# Magenta RT — Reverb × Low‑Pass Filter

A fine‑tuned **Magenta RealTime (Magenta RT)** checkpoint + **Colab notebook** that streams audio generation and lets you **toggle Reverb + Low‑Pass Filter live** on the drum stem.

This project keeps **all control in lane 0** (no extra lanes) to avoid style/control entanglement.

---

## Quickstart (Colab)

1. Open `demo.ipynb` in Colab  
2. Run cells top → bottom  
3. Set `HF_REPO_ID` + `CHECKPOINT_SUBDIR` in the “Load checkpoint” cell  
4. Click **Start** in the streaming widget  
5. Use **Reverb** + **LPF** toggles while it plays (applies on the next ~2s chunk)  
6. Toggle **record** → click **save wav** to download your recording

---

## Control scheme (single lane)

We use the **unused vocab gap** (after codec vocab, before style vocab):

- `VOCAB_CONTROL_OFFSET = vocab_codec_offset + vocab_codec_size`
- `control_token = VOCAB_CONTROL_OFFSET + state_id`
- `state_id = reverb_id * 4 + lpf_id`

Where:

- `reverb_id`: `0=dry`, `1=light`, `2=medium`, `3=heavy`
- `lpf_id`: `0=open`, `1=light`, `2=medium`, `3=heavy`

### Why lane‑0 only?

Earlier experiments that used multiple lanes for control got **entangled** with style lanes.  
Lane‑0-only control (and keeping lanes 1–5 as real MusicCoCa style tokens) stays much cleaner.

---

## System patch (required)

The notebook applies a small patch to `magenta_rt/system.py`:

- After `style_tokens_lm` is computed, we do:
  - `style_tokens_lm[0] = self._control_lane0_token` (if set)

This means:

- Lane 0 → control token (our state)
- Lanes 1–5 → normal MusicCoCa style tokens (matches training)

---

## 🧪 Alternative attempted approach (token appending)

We also tried an approach where control tokens were **appended** to the prompt sequence (extra tokens after context/style).  
It did **not** work as well as lane‑0 override: the model tended to ignore the appended controls or behave inconsistently.

---

## 🏋️ Training data (high level)

Training used a mixture of:

- **Slakh2100** (audio files + our own renderings based on MIDI)
- **MUSDB** and other **stem‑separated** music
- FX augmentation via **Pedalboard** (reverb + low‑pass filter variants)

---

## Repo contents

- `demo.ipynb` — Colab notebook (install → patch → load checkpoint → live UI)
- `requirements.txt` — minimal local deps (Colab uses install cells)

---

## License

- Upstream Magenta RT code: **Apache 2.0**
- Upstream open weights: **CC BY 4.0**

If you distribute checkpoints, include attribution consistent with upstream licensing.
