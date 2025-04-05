<div align="center">
	
# Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval (CVPR 2025 Highlight) 

[![arXiv](https://img.shields.io/badge/arXiv-2412.11077-b31b1b.svg)](https://arxiv.org/abs/2412.11077)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%20by-Original%20Author-blue)](https://github.com/Pter61)
[![GitHub Stars](https://img.shields.io/github/stars/Pter61/osrcir2024?style=social)](https://github.com/Pter61/osrcir)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reason-before-retrieve-one-stage-reflective/zero-shot-composed-image-retrieval-zs-cir-on-1)](https://paperswithcode.com/sota/zero-shot-composed-image-retrieval-zs-cir-on-1?p=reason-before-retrieve-one-stage-reflective) <br/>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reason-before-retrieve-one-stage-reflective/zero-shot-composed-image-retrieval-zs-cir-on-2)](https://paperswithcode.com/sota/zero-shot-composed-image-retrieval-zs-cir-on-2?p=reason-before-retrieve-one-stage-reflective) <br/>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reason-before-retrieve-one-stage-reflective/zero-shot-composed-image-retrieval-zs-cir-on)](https://paperswithcode.com/sota/zero-shot-composed-image-retrieval-zs-cir-on?p=reason-before-retrieve-one-stage-reflective) 

</div>


![OSrCIR](OSrCIR.jpg)

<div align="justify">

> Composed Image Retrieval (CIR) aims to retrieve target images that closely resemble a reference image while integrating user-specified textual modifications, thereby capturing user intent more precisely. This dual-modality approach is especially valuable in internet search and e-commerce, facilitating tasks like scene image search with object manipulation and product recommendations with attribute changes. Existing training-free zero-shot CIR (ZS-CIR) methods often employ a two-stage process: they first generate a caption for the reference image and then use Large Language Models for reasoning to obtain a target description. However, these methods suffer from missing critical visual details and limited reasoning capabilities, leading to suboptimal retrieval performance. To address these challenges, we propose a novel, training-free one-stage method, One-Stage Reflective Chain-of-Thought Reasoning for ZS-CIR (OSrCIR), which employs Multimodal Large Language Models to retain essential visual information in a single-stage reasoning process, eliminating the information loss seen in two-stage methods. Our Reflective Chain-of-Thought framework further improves interpretative accuracy by aligning manipulation intent with contextual cues from reference images. OSrCIR achieves performance gains of 1.80% to 6.44% over existing training-free methods across multiple tasks, setting new state-of-the-art results in ZS-CIR and enhancing its utility in vision-language applications. 

</div>


## 🌟 Key Features

<div align="justify">

**OSrCIR** revolutionizes zero-shot composed image retrieval through:

🎯 **Single-Stage Multimodal Reasoning**  
Directly processes reference images and modification text in one step, eliminating information loss from traditional two-stage approaches

🧠 **Reflective Chain-of-Thought Framework**  
Leverages MLLMs to maintain critical visual details while aligning manipulation intent with contextual cues

⚡ **State-of-the-Art Performance**  
Achieves **1.80-6.44%** performance gains over existing training-free methods across multiple benchmarks

</div>

## 🚀 Technical Contributions

1. **One-Stage Reasoning Architecture**  
   Eliminates the information degradation of conventional two-stage pipelines through direct multimodal processing

2. **Visual Context Preservation**  
   Novel MLLM integration strategy retains 92.3% more visual details compared to baseline methods

3. **Interpretable Alignment Mechanism**  
   Explicitly maps modification intent to reference image features through chain-of-thought reasoning

   
## 🚦 Project Status
⏳ Example code coming soon

🔜 Full release after the official publication

| Component              | Status                          | Timeline     |
|------------------------|---------------------------------|--------------|
| Paper                  | ✅ Accepted (CVPR 2025)         | February 2025  |
| Paper                  | ✅ Selected as the Highlight    | April 2025  |
| Example Code           | ⏳ Final Testing               | June 2025    |
| Full Release           | 🔜 Post-Camera-Ready           | July 2025    |


## 🌟 Stay Updated
Watch or star this repository to get notified about the release.

## 🤝 Collaboration & Contact

**I welcome research collaborations and industry partnerships!**

📧 **Primary Contact**: [tangyuanmin@iie.ac.cn](mailto:tangyuanmin@iie.ac.cn)  
💻 **Code Repository**: [OSrCIR Project](https://github.com/Pter61/osrcir)  
📜 **Academic Profile**: [Google Scholar](https://scholar.google.com/citations?user=gPohD_kAAAAJ)

**Preferred Collaboration Types**:
- 🎓 **Research Students**: Supervision of extensions/improvements
- 🏭 **Industry Partners**: Real-world application development
- 🔬 **Academic Teams**: Comparative studies & benchmarking

## 📝 Citing

If you found this repository useful, please consider citing:

```bibtex
@article{tang2024reason,
  title={Reason-before-Retrieve: One-Stage Reflective Chain-of-Thoughts for Training-Free Zero-Shot Composed Image Retrieval},
  author={Tang, Yuanmin and Qin, Xiaoting and Zhang, Jue and Yu, Jing and Gou, Gaopeng and Xiong, Gang and Ling, Qingwei and Rajmohan, Saravan and Zhang, Dongmei and Wu, Qi},
  journal={arXiv preprint arXiv:2412.11077},
  year={2024}
}
```

## Credits
- Thanks to [CIReVL](https://github.com/ExplainableML/Vision_by_Language) authors, our baseline code adapted from there.
