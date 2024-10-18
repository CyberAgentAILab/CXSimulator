<div align="center">
<h1>CXSimulator: A User Behavior Simulation using LLM Embeddings for Web-Marketing Campaign Assessment</h3>

<p align="center">
    <a href="https://sites.google.com/view/akira-kasuga/home">Akira Kasuga</a> &nbsp;
    <a href="https://yonetaniryo.github.io/">Ryo Yonetani</a> &nbsp;
</p>

<p align="center">
    CyberAgent, Inc. &nbsp;
</p>

<p align="center">
    <strong>CIKM 2024</strong>
</p>

<p align="center">
    <a href="https://arxiv.org/pdf/2407.21553"><img src="https://img.shields.io/badge/arXiv-paper-orange" alt="arXiv paper"></a>
    <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
</p>

</div>

<img src="https://github.com/user-attachments/assets/d24d85d8-e8ef-411e-918e-3230f31c6167" alt="Cover Image">

---

## 📌 Overview

**CXSimulator** framework uses LLMs to represent user behavior events as semantic embeddings and predicts transitions between these events. This enables simulation of user reactions to new campaigns, eliminating the need for costly online testing and providing valuable insights to marketers.

## 🛠 Prerequisites

| Operating System               | Based on                                                |
| ------------------------------ | ------------------------------------------------------- |
| Debian GNU/Linux 12 (bookworm) | [python:3.10-bookworm](https://hub.docker.com/_/python) |

| Software              | Install                                                                             |
| --------------------- | ----------------------------------------------------------------------------------- |
| Python >= 3.10,< 3.12 | -                                                                                   |
| [Poetry] >= 1.8.0     | [installer](https://python-poetry.org/docs/#installing-with-the-official-installer) |
| [pre-commit] >= 3.8.0 | `pip install pre-commit`                                                            |

| Cloud Infrastructure | Link                                                                                                                        | Summary                                                                                                                                                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cloud BigQuery       | [Google Analytics Sample](https://console.cloud.google.com/marketplace/product/obfuscated-ga360-data/obfuscated-ga360-data) | The dataset provides 12 months (August 2016 to August 2017) of obfuscated Google Analytics 360 data from the Google Merchandise Store , a real ecommerce store that sells Google-branded merchandise,                           |
| AzureOpenAI          | [Generate embeddings with Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/embeddings)       | An embedding is a special format of data representation that can be easily utilized by machine learning models and algorithms. The embedding is an information dense representation of the semantic meaning of a piece of text. |

## 🔧 Setup

```shell
poetry install
```

## 🚀 Getting started (Using Cache Data)

### Help

```shell
poetry run python -m cxsim --help
poetry run task --list
```

### Preprocess and Train

```shell
poetry run task model_using_cache
```

### Simulation

```shell
poetry run task simulation_using_cache
```

## 📊 Execute All Steps

### Environment Setting

> [!IMPORTANT]
> Authentication for cloud services is a prerequisite for executing all steps and may incur some costs.

#### Google Cloud

1. Enable BigQuery API in your project.

2. Install [gcloud CLI](https://cloud.google.com/sdk/docs/install)

3. Auth Login.

```shell
gcloud auth application-default login
```

#### Microsoft AzureOpenAI

1. Copy template
   ```bash
   cp ./src/cxsim/config/.env.template ./src/cxsim/config/.env
   ```
2. Add the following content to the `.env` file:

   ```bash
    # Azure OpenAI
    AZURE_OPENAI_US_ENDPOINT=XXXXXXXX
    AZURE_OPENAI_US_VERSION=2024-03-01-preview
    AZURE_OPENAI_US_KEY=XXXXXXXX
    # Google Cloud
    GOOGLE_CLOUD_PROJECT_ID=XXXXXXXX
   ```

### Preprocess and Train

> [!NOTE]
> Once you've completed `poetry run task model_using_cache`, you can skip this step. In the next step, you'll simulate your campaigns using pre-trained models.

```shell
poetry run task model
```

### Simulation

```shell
poetry run task simulation --campaign-title "Enjoy 1 month Free of YouTube Premium for Youtube related Product"
```

If you would like to new data period,

```shell
poetry run task simulation_for_new --campaign-title "Enjoy 1 month Free of YouTube Premium for Youtube related Product"
```

## 📄 Citation

```bibtex
@inproceedings{kasuga2024CXSimulator
  title={CXSimulator: A User Behavior Simulation using LLM Embeddings for Web-Marketing Campaign Assessment},
  author={Akira Kasuga and Ryo Yonetani},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and
Knowledge Management (CIKM ’24)},
  year={2024},
  url={https://github.com/CyberAgentAILab/CXSimulator.git},
  doi={https://doi.org/10.1145/3627673.3679894}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

[poetry]: https://python-poetry.org/
[pre-commit]: https://pre-commit.com/
