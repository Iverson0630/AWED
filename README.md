# AWED: Asymmetric Wavelet Encoder-Decoder
Framework for Simultaneous Gas Distribution
Mapping and Gas Source Localization
Gas distribution mapping (GDM) and gas source localization (GSL) are two crucial research areas in gas monitoring.
However, due to the time-varying and non-uniform nature of gas distribution and the limitations of gas sensors, the accurate
and rapid estimation of gas distribution from sparse sensor data is a challenging task. We view the GDM super-resolution task as the combination of image inpainting and image super-resolution and  propose an end-to-end model called asymmetric wavelet encoder-decoder (AWED) to address GDM and GSL from ultra-sparse sensor data. The model uses a simplified encoder and enhanced decoder, incorporating wavelet reconstruction module (WRM) to decode from both spatial and frequency domains. Additionally, a wavelet L1 loss is introduced to promote frequency domain similarity between predicted and real images. 

<img src="pic\model.png" alt="model" style="zoom:67%;" />

In real-world scenarios, there are typically two methods for collecting gas distribution data: one is through an array of sensors at fixed locations, and the other is using mobile robots or drones equipped with gas sensors to collect gas concentrations along an S-shaped trajectory. Therefore, we design two sampling strategies: grid sampling strategy and S-shaped sampling strategy.

<img src="pic\sampling_strategy.png" alt="sampling_strategy" style="zoom: 50%;" />



The proposed method achieves a 32x super-resolution of gas distribution maps from 7x7 sensor data to 224x224 resolution images, and achieves a gas source localization accuracy of 0.32m within a 10mx10m area. The model also exhibits fewer parameters, faster prediction speed, and better real-time performance compared to existing  deep learning methods.

<img src="pic\result.png" alt="result" style="zoom:67%;" />

<img src="pic\result2.png" alt="result2" style="zoom: 67%;" />
