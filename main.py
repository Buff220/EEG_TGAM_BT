"""
TGAM multi-window detector (single forehead electrode, Bluetooth serial).

Install:  pip install pyserial numpy scipy pyqtgraph PyQt5
Run:      python tgam_multi.py COM10 57600
Test w/o hardware: python tgam_multi.py --demo

Windows:
  1. Overview   - raw trace, spectrogram, attention/meditation, signal quality
  2. Blink      - single / double / triple blink
  3. Muscle     - jaw CLENCH (sustained) vs short BURST (tongue/swallow/twitch)
  4. Alpha      - eyes CLOSED vs OPEN (8-12 Hz power)
  5. Eye move   - UP / DOWN (experimental)

Thresholds are the dashed horizontal lines: DRAG them with the mouse, or press "Auto".
"""
import argparse
import collections
import struct
import threading

import numpy as np
import pyqtgraph as pg
import serial
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy import signal

ap = argparse.ArgumentParser()
ap.add_argument("port", nargs="?", default="COM10")
ap.add_argument("baud", nargs="?", type=int, default=57600)
ap.add_argument("--mains", type=float, default=50.0, help="mains frequency to notch (50 or 60)")
ap.add_argument("--demo", action="store_true", help="synthetic data, no hardware")
ap.add_argument("--run-seconds", type=float, default=0, help="auto-quit (testing)")
args = ap.parse_args()

FS = 512
DOUBLE_WIN = 0.7      # s: max gap between blinks of a double blink
MERGE = 0.4           # s: weak rebound lobes after an eye event are ignored
BURST_MAX = 0.6       # s: muscle activity shorter than this = BURST, longer = CLENCH

q = collections.deque()                  # raw samples from the device
state = {"poor": 200, "att": 0, "med": 0, "packets": 0, "bad": 0}
stop = threading.Event()


# ============================ ThinkGear parser ============================
def parse_payload(p):
    i = 0
    while i < len(p):
        while i < len(p) and p[i] == 0x55:
            i += 1
        if i >= len(p):
            break
        code = p[i]
        if code >= 0x80:
            if i + 1 >= len(p):
                break
            n = p[i + 1]
            val = p[i + 2:i + 2 + n]
            i += 2 + n
            if code == 0x80 and n == 2:
                q.append(struct.unpack(">h", bytes(val))[0])
        else:
            if i + 1 >= len(p):
                break
            v = p[i + 1]
            i += 2
            if code == 0x02:
                state["poor"] = v
            elif code == 0x04:
                state["att"] = v
            elif code == 0x05:
                state["med"] = v


def reader():
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except Exception as e:
        print(f"Could not open {args.port}: {e}")
        return
    print(f"Connected to {args.port} @ {args.baud}")
    buf = bytearray()
    while not stop.is_set():
        try:
            d = ser.read(ser.in_waiting or 1)
        except Exception as e:
            print("Serial error:", e)
            break
        buf.extend(d)
        while True:
            idx = buf.find(b"\xaa\xaa")
            if idx < 0:
                buf = buf[-1:]
                break
            del buf[:idx]
            if len(buf) < 4:
                break
            plen = buf[2]
            if plen == 0xAA:
                del buf[0]
                continue
            if plen > 169:
                del buf[:2]
                continue
            if len(buf) < 4 + plen:
                break
            pay = bytes(buf[3:3 + plen])
            if ((~sum(pay)) & 0xFF) == buf[3 + plen]:
                state["packets"] += 1
                parse_payload(pay)
                del buf[:4 + plen]
            else:
                state["bad"] += 1
                del buf[:2]
    ser.close()


def demo():
    """Synthetic signal: blinks, double blink, jaw clench, tongue burst, alpha, eye-up."""
    import time
    rng = np.random.default_rng(1)
    n = 0
    pulse = lambda t, c, w, a: a * np.exp(-((t - c) / w) ** 2)
    state["poor"] = 0
    while not stop.is_set():
        t = (n + np.arange(32)) / FS
        n += 32
        tt = t % 20
        x = rng.normal(0, 25, 32) + 30 * np.sin(2 * np.pi * 10 * t) * (1 + 3 * ((tt > 13) & (tt < 17)))
        x += pulse(tt, 2, .08, 900) + pulse(tt, 5, .08, 900) + pulse(tt, 5.5, .08, 900)
        x += rng.normal(0, 120, 32) * ((tt > 8) & (tt < 9.5)) + rng.normal(0, 120, 32) * ((tt > 11) & (tt < 11.25))
        x += 300 * (pulse(tt, 18, .06, 1) - .6 * pulse(tt, 18.25, .08, 1))
        q.extend(int(v) for v in x)
        time.sleep(32 / FS)


# ============================ DSP helpers ============================
class Filt:
    def __init__(self, sos):
        self.sos = sos
        self.zi = np.zeros((sos.shape[0], 2))

    def __call__(self, x):
        y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
        return y


class Ring:
    def __init__(self, n):
        self.a = np.zeros(n)

    def push(self, x):
        n = len(x)
        if n >= len(self.a):
            self.a[:] = x[-len(self.a):]
        else:
            self.a = np.roll(self.a, -n)
            self.a[-n:] = x


notch = Filt(signal.tf2sos(*signal.iirnotch(args.mains, 30, FS)))
band_eye = Filt(signal.butter(2, [0.5, 10], "bandpass", fs=FS, output="sos"))
hp_emg = Filt(signal.butter(4, 30, "highpass", fs=FS, output="sos"))
env_lp = Filt(signal.butter(2, 4, "lowpass", fs=FS, output="sos"))

r_raw, r_eye, r_env = Ring(FS * 4), Ring(FS * 4), Ring(FS * 4)
r_long = Ring(FS * 2)  # for FFT


# ============================ GUI helpers ============================
app = QtWidgets.QApplication([])
pg.setConfigOptions(antialias=False)
clock = {"n": 0}


class Panel(QtWidgets.QWidget):
    def __init__(self, title, x, y, w, h):
        super().__init__()
        self.setWindowTitle(title)
        self.move(x, y)
        self.resize(w, h)
        self.flash_until = 0.0
        v = QtWidgets.QVBoxLayout(self)
        self.big = QtWidgets.QLabel()
        self.big.setAlignment(QtCore.Qt.AlignCenter)
        self.set_big("waiting...", "#888")
        v.addWidget(self.big)
        self.glw = pg.GraphicsLayoutWidget()
        v.addWidget(self.glw, 1)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        self.row = QtWidgets.QHBoxLayout()
        v.addLayout(self.row)
        self.show()

    def set_big(self, text, color):
        self.big.setText(text)
        self.big.setStyleSheet(f"font-size:30px;font-weight:bold;color:{color};padding:6px;")

    def flash(self, text, color, t, hold=2.0):
        self.set_big(text, color)
        self.flash_until = t + hold

    def idle(self, text, color, t):
        if t >= self.flash_until:
            self.set_big(text, color)

    def button(self, text, fn):
        b = QtWidgets.QPushButton(text)
        b.clicked.connect(fn)
        self.row.addWidget(b)
        return b


def hline(plot, val, color, label):
    ln = pg.InfiniteLine(pos=val, angle=0, movable=True, pen=pg.mkPen(color, style=QtCore.Qt.DashLine, width=2),
                         label=label, labelOpts={"position": 0.95, "color": color})
    plot.addItem(ln)
    return ln


# ---- window 1: overview
w_ov = Panel("1. Overview", 0, 0, 900, 720)
p = w_ov.glw.addPlot(row=0, col=0, title="Raw EEG (mains-notched)")
p.setYRange(-1500, 1500)
p.showGrid(x=True, y=True, alpha=.3)
c_raw = p.plot(pen="c")
t4 = np.linspace(-4, 0, FS * 4)
p = w_ov.glw.addPlot(row=1, col=0, title="Spectrogram 0-60 Hz (last 20 s)  |  blink = low freq, jaw = high freq, alpha = ~10 Hz")
SG_COLS = 80
sg = np.full((SG_COLS, 61), -0.5)
img = pg.ImageItem()
try:
    img.setLookupTable(pg.colormap.get("viridis").getLookupTable(nPts=256))
except Exception:
    pass
img.setRect(QtCore.QRectF(-20, 0, 20, 60))
img.setLevels((-0.5, 2.0))
p.addItem(img)
p.setLabel("left", "Hz")
p = w_ov.glw.addPlot(row=2, col=0, title="Attention (red) / Meditation (green)")
p.setYRange(0, 100)
c_att, c_med = p.plot(pen=pg.mkPen("r", width=2)), p.plot(pen=pg.mkPen("g", width=2))
att_h, med_h = collections.deque([0] * 120, maxlen=120), collections.deque([0] * 120, maxlen=120)

# ---- window 2: blink
w_bl = Panel("2. Blink detector", 910, 0, 700, 360)
p_bl = w_bl.glw.addPlot(title="Eye band 0.5-10 Hz (drag thresholds)")
p_bl.setYRange(-1200, 1200)
p_bl.showGrid(x=True, y=True, alpha=.3)
c_eye = p_bl.plot(pen="y")
ln_blink = hline(p_bl, 500, "r", "blink")
ln_eye = hline(p_bl, 150, "c", "eye move")
blink_t, blink_cnt = [], collections.Counter()

# ---- window 3: muscle
w_mu = Panel("3. Muscle: jaw / tongue", 910, 380, 700, 360)
p_mu = w_mu.glw.addPlot(title="Muscle envelope (30+ Hz energy)")
p_mu.showGrid(x=True, y=True, alpha=.3)
c_env = p_mu.plot(pen="m")
ln_emg = hline(p_mu, 40, "r", "threshold")
emg = {"active": False, "start": 0.0, "sustained": False}
emg_cnt = collections.Counter()

# ---- window 4: alpha
w_al = Panel("4. Alpha: eyes open / closed", 0, 740, 900, 330)
p_ar = w_al.glw.addPlot(row=0, col=0, title="Alpha ratio (8-12 Hz / 2-40 Hz)")
p_ar.setYRange(0, 1)
c_ar = p_ar.plot(pen=pg.mkPen("g", width=2))
ln_alpha = hline(p_ar, 0.25, "y", "closed >")
p_as = w_al.glw.addPlot(row=0, col=1, title="Spectrum")
p_as.setXRange(0, 40)
c_as = p_as.plot(pen="w")
p_as.addItem(pg.LinearRegionItem((8, 12), movable=False, brush=(0, 255, 0, 40)))
fx = np.fft.rfftfreq(FS * 2, 1 / FS)
hann = np.hanning(FS * 2)
ar_h = collections.deque(maxlen=240)
ar_base = {}

# ---- window 5: eye move
w_ey = Panel("5. Eye movement UP / DOWN (experimental)", 1620, 0, 520, 360)
w_ey.info.setText("Looks at spikes SMALLER than a blink. Flip polarity if UP/DOWN are swapped.")
cb_inv = QtWidgets.QCheckBox("Invert polarity")
w_ey.row.addWidget(cb_inv)
eye_cnt, eye_log = collections.Counter(), collections.deque(maxlen=6)


# ============================ detectors ============================
class EventExtractor:
    """Finds eye-band deflections above the low threshold. An event ends at the zero crossing;
    the smaller rebound lobes that follow (filter undershoot) within MERGE seconds are ignored."""
    def __init__(self):
        self.active = False
        self.start = self.peak = 0.0
        self.sign = 1
        self.last_peak, self.last_sign, self.last_end = 0.0, 0, -10.0

    def feed(self, x, t0, thr):
        out = []
        for i, v in enumerate(x):
            t = t0 + i / FS
            a = abs(v)
            sg = 1 if v > 0 else -1
            if not self.active:
                rebound = t - self.last_end < MERGE and (sg != self.last_sign or a < 0.6 * self.last_peak)
                if a > thr and not rebound:
                    self.active, self.start, self.peak, self.sign = True, t, a, sg
            else:
                if a > self.peak and sg == self.sign:
                    self.peak = a
                if sg != self.sign or a < 0.3 * self.peak:
                    out.append((self.start, self.peak, self.sign))
                    self.active, self.last_peak, self.last_sign, self.last_end = False, self.peak, self.sign, t
        return out


extractor = EventExtractor()
last_event_t = -10.0
last_fft = 0
calibrated = False


def auto_thresholds():
    noise = 1.4826 * np.median(np.abs(r_eye.a))
    lo = max(5 * noise, 120)
    ln_eye.setValue(lo)
    ln_blink.setValue(max(3 * lo, 450))
    ln_emg.setValue(max(4 * np.median(r_env.a), 15))


w_bl.button("Auto thresholds", auto_thresholds)
w_mu.button("Auto threshold", auto_thresholds)


def set_alpha_base(kind):
    vals = [v for v in list(ar_h)[-30:] if v is not None]
    if not vals:
        return
    ar_base[kind] = float(np.mean(vals))
    if len(ar_base) == 2:
        ln_alpha.setValue((ar_base["open"] + ar_base["closed"]) / 2)
        w_al.info.setText(f"open={ar_base['open']:.2f}  closed={ar_base['closed']:.2f}  -> threshold set")
    else:
        w_al.info.setText(f"{kind} baseline saved ({ar_base[kind]:.2f}); now do the other one")


w_al.button("Calibrate: eyes OPEN (last 8 s)", lambda: set_alpha_base("open"))
w_al.button("Calibrate: eyes CLOSED (last 8 s)", lambda: set_alpha_base("closed"))


def tick():
    global last_event_t, last_fft, calibrated
    batch = []
    while True:
        try:
            batch.append(q.popleft())
        except IndexError:
            break
    poor = state["poor"]
    q_txt = "GOOD" if poor == 0 else ("NO CONTACT" if poor == 200 else f"poor ({poor})")
    w_ov.info.setText(f"signal: {q_txt} | packets: {state['packets']} | bad checksums: {state['bad']} | "
                      f"attention {state['att']} meditation {state['med']}")
    if not batch:
        return
    x = notch(np.array(batch, float))
    t0 = clock["n"] / FS
    clock["n"] += len(x)
    t = clock["n"] / FS
    eye = band_eye(x)
    env = env_lp(np.abs(hp_emg(x)))
    for r, d in ((r_raw, x), (r_eye, eye), (r_env, env), (r_long, x)):
        r.push(d)
    if not calibrated and t > 4:
        auto_thresholds()
        calibrated = True

    # ---- muscle
    thr = ln_emg.value()
    for i, e in enumerate(env):
        ti = t0 + i / FS
        if not emg["active"] and e > thr:
            emg["active"], emg["start"] = True, ti
        elif emg["active"] and e < 0.6 * thr:
            emg["active"] = False
            d = ti - emg["start"]
            if d >= BURST_MAX:
                emg_cnt["clench"] += 1
                w_mu.flash(f"JAW CLENCH ({d:.1f}s)", "#e53935", ti)
                print(f"[{ti:6.1f}s] JAW CLENCH {d:.1f}s")
            else:
                emg_cnt["burst"] += 1
                w_mu.flash(f"SHORT BURST ({d:.2f}s)", "#fb8c00", ti)
                print(f"[{ti:6.1f}s] MUSCLE BURST (tongue/swallow/twitch) {d:.2f}s")
    emg["sustained"] = emg["active"] and (t - emg["start"]) > 0.25
    if emg["active"] and t - emg["start"] >= BURST_MAX:
        w_mu.flash("JAW CLENCH...", "#e53935", t, 0.3)
    w_mu.idle("RELAXED", "#43a047", t)
    w_mu.info.setText(f"clenches: {emg_cnt['clench']}   bursts: {emg_cnt['burst']}")

    # ---- eye events (blink / eye move)
    for (te, peak, sign) in extractor.feed(eye, t0, ln_eye.value()):
        last_event_t = te
        if emg["sustained"]:
            continue
        if peak >= ln_blink.value():
            blink_t.append(te)
        else:
            up = (sign > 0) != cb_inv.isChecked()
            name = "UP" if up else "DOWN"
            eye_cnt[name] += 1
            eye_log.appendleft(f"{te:6.1f}s  {name}  (peak {peak:.0f})")
            w_ey.flash(f"LOOK {name}", "#1e88e5" if up else "#8e24aa", te, 1.5)
            print(f"[{te:6.1f}s] EYE {name} peak={peak:.0f}")
    if blink_t and t - blink_t[-1] > DOUBLE_WIN:
        n = min(len(blink_t), 3)
        label = {1: "SINGLE BLINK", 2: "DOUBLE BLINK", 3: "TRIPLE BLINK"}[n]
        blink_cnt[label] += 1
        w_bl.flash(label, ["", "#43a047", "#fb8c00", "#e53935"][n], t, 2.0)
        print(f"[{t:6.1f}s] {label}")
        blink_t.clear()
    w_bl.idle("no blink", "#888", t)
    w_bl.info.setText("  ".join(f"{k}: {v}" for k, v in blink_cnt.items()) or "blink hard to test")
    w_ey.idle("-", "#888", t)
    w_ey.info.setText(f"UP: {eye_cnt['UP']}  DOWN: {eye_cnt['DOWN']}    " + " | ".join(eye_log))

    # ---- FFT stuff (4x per second)
    if clock["n"] - last_fft >= FS // 4:
        last_fft = clock["n"]
        seg = r_long.a - r_long.a.mean()
        spec = np.abs(np.fft.rfft(seg * hann)) / len(seg)
        c_as.setData(fx, spec)
        pw = spec ** 2
        ratio = pw[(fx >= 8) & (fx <= 12)].sum() / (pw[(fx >= 2) & (fx <= 40)].sum() + 1e-9)
        artifact = emg["sustained"] or (t - last_event_t) < 1.5
        if artifact:
            ar_h.append(None)
            w_al.idle("artifact (blink/muscle)", "#888", t)
        else:
            ar_h.append(ratio)
            closed = ratio > ln_alpha.value()
            w_al.idle("EYES CLOSED / relaxed" if closed else "EYES OPEN", "#43a047" if closed else "#1e88e5", t)
        y = np.array([np.nan if v is None else v for v in ar_h], float)
        c_ar.setData(y)
        # spectrogram column (1 s window)
        s1 = r_long.a[-FS:]
        col = np.log10(np.abs(np.fft.rfft((s1 - s1.mean()) * np.hanning(FS))) / FS + 1e-3)[:61]
        sg[:-1] = sg[1:]
        sg[-1] = col
        img.setImage(sg, autoLevels=False)
        att_h.append(state["att"])
        med_h.append(state["med"])
        c_att.setData(list(att_h))
        c_med.setData(list(med_h))

    # ---- plots
    c_raw.setData(t4, r_raw.a)
    c_eye.setData(t4, r_eye.a)
    c_env.setData(t4, r_env.a)


timer = QtCore.QTimer()
timer.timeout.connect(tick)
timer.start(33)
threading.Thread(target=demo if args.demo else reader, daemon=True).start()
if args.run_seconds:
    QtCore.QTimer.singleShot(int(args.run_seconds * 1000), app.quit)
try:
    pg.exec()
finally:
    stop.set()
