# SignSense: Demo Examples for Member 3 (Adversarial Robustness)
These test sample images illustrate where the **Custom CNN** is successfully fooled by FGSM noise, but the **Robust CNN (Adversarial Training)** successfully resists and predicts the correct class.
| Sample Image | True / Clean Class | Epsilon (Noise Strength) | Standard CNN Prediction | Robust CNN Prediction | Defense Status |
|---|---|---|---|---|---|
| `samples/1_speed_30.png` | **Speed limit (30km/h)** | `0.04` | ❌ Speed limit (120km/h) | ✅ **Speed limit (30km/h)** | **Resisted** |
| `samples/28_children_crossing.png` | **Children crossing** | `0.03` | ❌ Pedestrians | ✅ **Children crossing** | **Resisted** |
| `samples/2_speed_50.png` | **Speed limit (50km/h)** | `0.02` | ❌ Speed limit (80km/h) | ✅ **Speed limit (50km/h)** | **Resisted** |
| `samples/40_roundabout.png` | **Roundabout mandatory** | `0.04` | ❌ Keep right | ✅ **Roundabout mandatory** | **Resisted** |
| `samples/5_speed_80.png` | **Speed limit (80km/h)** | `0.05` | ❌ Speed limit (20km/h) | ✅ **Speed limit (80km/h)** | **Resisted** |
