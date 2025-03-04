1 arXiv:2407.21783v3 [cs.AI] 23 Nov 2024

The Llama 3 Herd of Models

Llama Team, AI @ Meta¹

¹A detailed contributor list can be found in the appendix of this paper.

Modern artificial intelligence (AI) systems are powered by foundation models. This paper presents a new set of foundation models, called Llama 3. It is a herd of language models that natively support multilinguality, coding, reasoning, and tool usage. Our largest model is a dense Transformer with 405B parameters and a context window of up to 128K tokens. This paper presents an extensive empirical evaluation of Llama 3. We find that Llama 3 delivers comparable quality to leading language models such as GPT-4 on a plethora of tasks. We publicly release Llama 3, including pre-trained and post-trained versions of the 405B parameter language model and our Llama Guard 3 model for input and output safety. The paper also presents the results of experiments in which we integrate image, video, and speech capabilities into Llama 3 via a compositional approach. We observe this approach performs competitively with the state-of-the-art on image, video, and speech recognition tasks. The resulting models are not yet being broadly released as they are still under development.

Date: July 23, 2024

Website: https://llama.meta.com/

1 Introduction

Foundation models are general models of language, vision, speech, and/or other modalities that are designed to support a large variety of AI tasks. They form the basis of many modern AI systems.

The development of modern foundation models consists of two main stages: (1) a pre-training stage in which the model is trained at massive scale using straightforward tasks such as next-word prediction or captioning and (2) a post-training stage in which the model is tuned to follow instructions, align with human preferences, and improve specific capabilities (for example, coding and reasoning).

In this paper, we present a new set of foundation models for language, called Llama 3. The Llama 3 Herd of models natively supports multilinguality, coding, reasoning, and tool usage. Our largest model is a dense Transformer with 405B parameters, processing information in a context window of up to 128K tokens. Each member of the herd is listed in Table 1. All the results presented in this paper are for the Llama 3.1 models, which we will refer to as Llama 3 throughout for brevity.

We believe there are three key levers in the development of high-quality foundation models: data, scale, and managing complexity. We seek to optimize for these three levers in our development process:

• Data. Compared to prior versions of Llama (Touvron et al., 2023a,b), we improved both the quantity and quality of the data we use for pre-training and post-training. These improvements include the development of more careful pre-processing and curation pipelines for pre-training data and the development of more rigorous quality assurance and filtering approaches for post-training data. We pre-train Llama 3 on a corpus of about 15T multilingual tokens, compared to 1.8T tokens for Llama 2.

• Scale. We train a model at far larger scale than previous Llama models: our flagship language model was pre-trained using 3.8 × 10<sup>25</sup> FLOPs, almost 50× more than the largest version of Llama 2. Specifically, we pre-trained a flagship model with 405B trainable parameters on 15.6T text tokens. As expected per

<table style="width:100%;">
<tr>
<th>Finetuned</th>
<th>Multilingual</th>
<th>Long context</th>
<th>Tool use</th>
<th>Release</th>
</tr>
<tr>
<td>Llama 3 8B</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>April 2024</td>
</tr>
<tr>
<td>Llama 3 8B Instruct</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>April 2024</td>
</tr>
<tr>
<td>Llama 3 70B</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>April 2024</td>
</tr>
<tr>
<td>Llama 3 70B Instruct</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>April 2024</td>
</tr>
<tr>
<td>Llama 3.1 8B</td>
<td>✗</td>
<td>✓</td>
<td>✓</td>
<td>✗</td>
<td>July 2024</td>
</tr>
<tr>
<td>Llama 3.1 8B Instruct</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>July 2024</td>
</tr>
<tr>
<td>Llama 3.1 70B</td>
<td>✗</td>
<td>✓</td>
<td>✓</td>
<td>✗</td>
<td>July 2024</td>
</tr>
<tr>
<td>Llama 3.1 70B Instruct</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>July 2024</td>
</tr>
<tr>
<td>Llama 3.1 405B</td>
<td>✗</td>
<td>✓</td>
<td>✓</td>
<td>✗</td>
<td>July 2024</td>
</tr>
<tr>
<td>Llama 3.1 405B Instruct</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>July 2024</td>
</tr>
</table>
<p>Table 1 Overview of the Llama 3 Herd of models. All results in this paper are for the Llama 3.1 models.</p>
<p>scaling laws for foundation models, our flagship model outperforms smaller models trained using the
same procedure. While our scaling laws suggest our flagship model is an approximately compute-optimal
size for our training budget, we also train our smaller models for much longer than is compute-optimal.
The resulting models perform better than compute-optimal models at the same inference budget. We
use the flagship model to further improve the quality of those smaller models during post-training.</p>
<p>•Managing complexity. We make design choices that seek to maximize our ability to scale the model
development process. For example, we opt for a standard dense Transformer model architecture (Vaswani
et al., 2017) with minor adaptations, rather than for a mixture-of-experts model (Shazeer et al., 2017)
to maximize training stability. Similarly, we adopt a relatively simple post-training procedure based
on supervised finetuning (SFT), rejection sampling (RS), and direct preference optimization (DPO;
Rafailov et al. (2023)) as opposed to more complex reinforcement learning algorithms (Ouyang et al.,
2022; Schulman et al., 2017) that tend to be less stable and harder to scale.</p>
<p>The result of our work is Llama 3: a herd of three multilingual<sup>1</sup> language models with 8B, 70B, and 405B
parameters. We evaluate the performance of Llama 3 on a plethora of benchmark datasets that span a wide
range of language understanding tasks. In addition, we perform extensive human evaluations that compare
Llama 3 with competing models. An overview of the performance of the flagship Llama 3 model on key
benchmarks is presented in Table 2. Our experimental evaluation suggests that our flagship model performs
on par with leading language models such as GPT-4 (OpenAI, 2023a) across a variety of tasks, and is close to
matching the state-of-the-art. Our smaller models are best-in-class, outperforming alternative models with
similar numbers of parameters (Bai et al., 2023; Jiang et al., 2023). Llama 3 also delivers a much better
balance between helpfulness and harmlessness than its predecessor (Touvron et al., 2023b). We present a
detailed analysis of the safety of Llama 3 in Section 5.4.</p>
<p>We are publicly releasing all three Llama 3 models under an updated version of the Llama 3 Community License;
see https://llama.meta.com. This includes pre-trained and post-trained versions of our 405B parameter
language model and a new version of our Llama Guard model (Inan et al., 2023) for input and output safety.
We hope that the open release of a flagship model will spur a wave of innovation in the research community,
and accelerate a responsible path towards the development of artificial general intelligence (AGI).</p>
<p>As part of the Llama 3 development process we also develop multimodal extensions to the models, enabling
image recognition, video recognition, and speech understanding capabilities. These models are still under
active development and not yet ready for release. In addition to our language modeling results, the paper
presents results of our initial experiments with those multimodal models.</p>
<p><sup>1</sup>The Llama 3 8B and 70B were pre-trained on multilingual data but were intended for use in English at the time.</p>
<p>2</p>
<table>
<thead>
<tr>
<th>Category</th>
<th>Benchmark</th>
<th>Llama 3 8B</th>
<th>Gemma 2 9B</th>
<th>Mistral 7B</th>
<th>Llama 3 70B</th>
<th>Mixtral 8x22B</th>
<th>GPT 3.5 Turbo</th>
<th>Llama 3 405B</th>
<th>Nemotron 4 340B</th>
<th>GPT-4 (0125)</th>
<th>GPT-4o</th>
<th>Claude 3.5 Sonnet</th>
</tr>
</thead>
<tbody>
<tr>
<td>General</td>
<td>MMLU (5-shot)</td>
<td>69.4</td>
<td>72.3</td>
<td>61.1</td>
<td>83.6</td>
<td>76.9</td>
<td>70.7</td>
<td>87.3</td>
<td>82.6</td>
<td>85.1</td>
<td>89.1</td>
<td>89.9</td>
</tr>
<tr>
<td></td>
<td>MMLU (0-shot, CoT)</td>
<td>73.0</td>
<td>72.3</td>
<td>60.5</td>
<td>86.0</td>
<td>79.9</td>
<td>69.8</td>
<td>88.6</td>
<td>78.7</td>
<td>85.4</td>
<td>88.7</td>
<td>88.3</td>
</tr>
<tr>
<td></td>
<td>MMLU-Pro (5-shot, CoT)</td>
<td>48.3</td>
<td>-</td>
<td>36.9</td>
<td>66.4</td>
<td>56.3</td>
<td>49.2</td>
<td>73.3</td>
<td>62.7</td>
<td>64.8</td>
<td>74.0</td>
<td>77.0</td>
</tr>
<tr>
<td></td>
<td>IFEval</td>
<td>80.4</td>
<td>73.6</td>
<td>57.6</td>
<td>87.5</td>
<td>72.7</td>
<td>69.9</td>
<td>88.6</td>
<td>85.1</td>
<td>84.3</td>
<td>85.6</td>
<td>88.0</td>
</tr>
<tr>
<td>Code</td>
<td>HumanEval (0-shot)</td>
<td>72.6</td>
<td>54.3</td>
<td>40.2</td>
<td>80.5</td>
<td>75.6</td>
<td>68.0</td>
<td>89.0</td>
<td>73.2</td>
<td>86.6</td>
<td>90.2</td>
<td>92.0</td>
</tr>
<tr>
<td></td>
<td>MBPP EvalPlus (0-shot)</td>
<td>72.8</td>
<td>71.7</td>
<td>49.5</td>
<td>86.0</td>
<td>78.6</td>
<td>82.0</td>
<td>88.6</td>
<td>72.8</td>
<td>83.6</td>
<td>87.8</td>
<td>90.5</td>
</tr>
<tr>
<td>Math</td>
<td>GSM8K (8-shot, CoT)</td>
<td>84.5</td>
<td>76.7</td>
<td>53.2</td>
<td>95.1</td>
<td>88.2</td>
<td>81.6</td>
<td>96.8</td>
<td>92.3</td>
<td>94.2</td>
<td>96.1</td>
<td>96.4</td>
</tr>
<tr>
<td></td>
<td>MATH (0-shot, CoT)</td>
<td>51.9</td>
<td>44.3</td>
<td>13.0</td>
<td>68.0</td>
<td>54.1</td>
<td>43.1</td>
<td>73.8</td>
<td>41.1</td>
<td>64.5</td>
<td>76.6</td>
<td>71.1</td>
</tr>
<tr>
<td>Reasoning</td>
<td>ARC Challenge (0-shot)</td>
<td>83.4</td>
<td>87.6</td>
<td>74.2</td>
<td>94.8</td>
<td>88.7</td>
<td>83.7</td>
<td>96.9</td>
<td>94.6</td>
<td>96.4</td>
<td>96.7</td>
<td>96.7</td>
</tr>
<tr>
<td></td>
<td>GPQA (0-shot, CoT)</td>
<td>32.8</td>
<td>-</td>
<td>28.8</td>
<td>46.7</td>
<td>33.3</td>
<td>30.8</td>
<td>51.1</td>
<td>-</td>
<td>41.4</td>
<td>53.6</td>
<td>59.4</td>
</tr>
<tr>
<td>Tool use</td>
<td>BFCL</td>
<td>76.1</td>
<td>-</td>
<td>60.4</td>
<td>84.8</td>
<td>-</td>
<td>85.9</td>
<td>88.5</td>
<td>86.5</td>
<td>88.3</td>
<td>80.5</td>
<td>90.2</td>
</tr>
<tr>
<td></td>
<td>Nexus</td>
<td>38.5</td>
<td>30.0</td>
<td>24.7</td>
<td>56.7</td>
<td>48.5</td>
<td>37.2</td>
<td>58.7</td>
<td>-</td>
<td>50.3</td>
<td>56.1</td>
<td>45.7</td>
</tr>
<tr>
<td>Long context</td>
<td>ZeroSCROLLS/QuALITY</td>
<td>81.0</td>
<td>-</td>
<td>-</td>
<td>90.5</td>
<td>-</td>
<td>-</td>
<td>95.2</td>
<td>-</td>
<td>95.2</td>
<td>90.5</td>
<td>90.5</td>
</tr>
<tr>
<td></td>
<td>InfiniteBench/En.MC</td>
<td>65.1</td>
<td>-</td>
<td>-</td>
<td>78.2</td>
<td>-</td>
<td>-</td>
<td>83.4</td>
<td>-</td>
<td>72.1</td>
<td>82.5</td>
<td>-</td>
</tr>
<tr>
<td></td>
<td>NIH/Multi-needle</td>
<td>98.8</td>
<td>-</td>
<td>-</td>
<td>97.5</td>
<td>-</td>
<td>-</td>
<td>98.1</td>
<td>-</td>
<td>100.0</td>
<td>100.0</td>
<td>90.8</td>
</tr>
<tr>
<td>Multilingual</td>
<td>MGSM (0-shot, CoT)</td>
<td>68.9</td>
<td>53.2</td>
<td>29.9</td>
<td>86.9</td>
<td>71.1</td>
<td>51.4</td>
<td>91.6</td>
<td>-</td>
<td>85.9</td>
<td>90.5</td>
<td>91.6</td>
</tr>
</tbody>
</table>
Figure 1 Illustration of the overall architecture and training of Llama 3. Llama 3 is a Transformer language model trained to predict the next token of a textual sequence. See text for details.

self-supervised approach that masks out parts of the speech inputs and tries to reconstruct the masked out parts via a discrete-token representation. As a result, the model learns the structure of speech signals. See Section 7 for details on the image encoder and Section 8 for details on the speech encoder.

• Vision adapter training. We train an adapter that integrates the pre-trained image encoder into the pre-trained language model. The adapter consists of a series of cross-attention layers that feed image-encoder representations into the language model. The adapter is trained on text-image pairs. This aligns the image representations with the language representations. During adapter training, we also update the parameters of the image encoder but we intentionally do not update the language-model parameters. We also train a video adapter on top of the image adapter on paired video-text data. This enables the model to aggregate information across frames. See Section 7 for details.

• Speech adapter training. Finally, we integrate the speech encoder into the model via an adapter that converts speech encodings into token representations that can be fed directly into the finetuned language model. The parameters of the adapter and encoder are jointly updated in a supervised finetuning stage to enable high-quality speech understanding. We do not change the language model during speech adapter training. We also integrate a text-to-speech system. See Section 8 for details.

Our multimodal experiments lead to models that can recognize the content of images and videos, and support interaction via a speech interface. These models are still under development and not yet ready for release.

3 Pre-Training

Language model pre-training involves: (1) the curation and filtering of a large-scale training corpus, (2) the development of a model architecture and corresponding scaling laws for determining model size, (3) the development of techniques for efficient pre-training at large scale, and (4) the development of a pre-training recipe. We present each of these components separately below.

3.1 Pre-Training Data

We create our dataset for language model pre-training from a variety of data sources containing knowledge until the end of 2023. We apply several de-duplication methods and data cleaning mechanisms on each data source to obtain high-quality tokens. We remove domains that contain large amounts of personally identifiable information (PII), and domains with known adult content.

3.1.1 Web Data Curation

Much of the data we utilize is obtained from the web and we describe our cleaning process below.

PII and safety filtering. Among other mitigations, we implement filters designed to remove data from websites that are likely to contain unsafe content or high volumes of PII, domains that have been ranked as harmful according to a variety of Meta safety standards, and domains that are known to contain adult content.

4

Text extraction and cleaning. We process the raw HTML content for non-truncated web documents to extract high-quality diverse text. To do so, we build a custom parser that extracts the HTML content and optimizes for precision in boilerplate removal and content recall. We evaluate our parser’s quality in human evaluations, comparing it with popular third-party HTML parsers that optimize for article-like content, and found it to perform favorably. We carefully process HTML pages with mathematics and code content to preserve the structure of that content. We maintain the image alt attribute text since mathematical content is often represented as pre-rendered images where the math is also provided in the alt attribute. We experimentally evaluate different cleaning configurations. We find markdown is harmful to the performance of a model that is primarily trained on web data compared to plain text, so we remove all markdown markers.

De-duplication. We apply several rounds of de-duplication at the URL, document, and line level:
• URL-level de-duplication. We perform URL-level de-duplication across the entire dataset. We keep the most recent version for pages corresponding to each URL.
• Document-level de-duplication. We perform global MinHash (Broder, 1997) de-duplication across the entire dataset to remove near duplicate documents.
• Line-level de-duplication. We perform aggressive line-level de-duplication similar to ccNet (Wenzek et al., 2019). We remove lines that appeared more than 6 times in each bucket of 30M documents. Although our manual qualitative analysis showed that the line-level de-duplication removes not only leftover boilerplate from various websites such as navigation menus, cookie warnings, but also frequent high-quality text, our empirical evaluations showed strong improvements.

Heuristic filtering. We develop heuristics to remove additional low-quality documents, outliers, and documents with excessive repetitions. Some examples of heuristics include:
• We use duplicated n-gram coverage ratio (Rae et al., 2021) to remove lines that consist of repeated content such as logging or error messages. Those lines could be very long and unique, hence cannot be filtered by line-dedup.
• We use “dirty word” counting (Raffel et al., 2020) to filter out adult websites that are not covered by domain block lists.
• We use a token-distribution Kullback-Leibler divergence to filter out documents containing excessive numbers of outlier tokens compared to the training corpus distribution.

Model-based quality filtering. Further, we experiment with applying various model-based quality classifiers to sub-select high-quality tokens. These include using fast classifiers such as fasttext (Joulin et al., 2017) trained to recognize if a given text would be referenced by Wikipedia (Touvron et al., 2023a), as well as more compute-intensive Roberta-based classifiers (Liu et al., 2019a) trained on Llama 2 predictions. To train a quality classifier based on Llama 2, we create a training set of cleaned web documents, describe the quality requirements, and instruct Llama 2’s chat model to determine if the documents meets these requirements. We use DistilRoberta (Sanh et al., 2019) to generate quality scores for each document for efficiency reasons. We experimentally evaluate the efficacy of various quality filtering configurations.

Code and reasoning data. Similar to DeepSeek-AI et al. (2024), we build domain-specific pipelines that extract code and math-relevant web pages. Specifically, both the code and reasoning classifiers are DistilRoberta models trained on web data annotated by Llama 2. Unlike the general quality classifier mentioned above, we conduct prompt tuning to target web pages containing math deduction, reasoning in STEM areas and code interleaved with natural language. Since the token distribution of code and math is substantially different than that of natural language, these pipelines implement domain-specific HTML extraction, customized text features and heuristics for filtering.

Multilingual data. Similar to our processing pipelines for English described above, we implement filters to remove data from websites that are likely to contain PII or unsafe content. Our multilingual text processing pipeline has several unique features:
• We use a fasttext-based language identification model to categorize documents into 176 languages.
• We perform document-level and line-level de-duplication within data for each language.

5

3.2 Model Architecture

Llama 3 uses a standard, dense Transformer architecture (Vaswani et al., 2017). It does not deviate significantly from Llama and Llama 2 (Touvron et al., 2023a,b) in terms of model architecture; our performance gains are primarily driven by improvements in data quality and diversity as well as by increased training scale.

We make a few small modifications compared to Llama 2:

• We use grouped query attention (GQA; Ainslie et al. (2023)) with 8 key-value heads to improve inference speed and to reduce the size of key-value caches during decoding.

• We use an attention mask that prevents self-attention between different documents within the same sequence. We find that this change had limited impact during standard pre-training, but find it to be important in continued pre-training on very long sequences.

6

<html>
<table border="1">
<tr>
<th></th>
<th>8B</th>
<th>70B</th>
<th>405B</th>
</tr>
<tr>
<td>Layers</td>
<td>32</td>
<td>80</td>
<td>126</td>
</tr>
<tr>
<td>Model Dimension</td>
<td>4,096</td>
<td>8192</td>
<td>16,384</td>
</tr>
<tr>
<td>FFN Dimension</td>
<td>14,336</td>
<td>28,672</td>
<td>53,248</td>
</tr>
<tr>
<td>Attention Heads</td>
<td>32</td>
<td>64</td>
<td>128</td>
</tr>
<tr>
<td>Key/Value Heads</td>
<td>8</td>
<td>8</td>
<td>8</td>
</tr>
<tr>
<td>Peak Learning Rate</td>
<td>3 × 10<sup>-4</sup></td>
<td>1.5 × 10<sup>-4</sup></td>
<td>8 × 10<sup>-5</sup></td>
</tr>
<tr>
<td>Activation Function</td>
<td>SwiGLU</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Vocabulary Size</td>
<td>128,000</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Positional Embeddings</td>
<td>ROPE (θ = 500,000)</td>
<td></td>
<td></td>
</tr>
</table>
<p>Table 3 Overview of the key hyperparameters of Llama 3. We display settings for 8B, 70B, and 405B language models.</p>
<p>• We use a vocabulary with 128K tokens. Our token vocabulary combines 100K tokens from the tiktoken<sup>3</sup> tokenizer with 28K additional tokens to better support non-English languages. Compared to the Llama 2 tokenizer, our new tokenizer improves compression rates on a sample of English data from 3.17 to 3.94 characters per token. This enables the model to "read" more text for the same amount of training compute. We also found that adding 28K tokens from select non-English languages improved both compression ratios and downstream performance, with no impact on English tokenization.</p>
<p>• We increase the RoPE base frequency hyperparameter to 500,000. This enables us to better support longer contexts; Xiong et al. (2023) showed this value to be effective for context lengths up to 32,768. Llama 3 405B uses an architecture with 126 layers, a token representation dimension of 16,384, and 128 attention heads; see Table 3 for details. This leads to a model size that is approximately compute-optimal according to scaling laws on our data for our training budget of 3.8 × 10<sup>25</sup> FLOPs.</p>
<p>3.2.1 Scaling Laws</p>
<p>We develop scaling laws (Hoffmann et al., 2022; Kaplan et al., 2020) to determine the optimal model size for our flagship model given our pre-training compute budget. In addition to determining the optimal model size, a major challenge is to forecast the flagship model’s performance on downstream benchmark tasks, due to a couple of issues: (1) Existing scaling laws typically predict only next-token prediction loss rather than specific benchmark performance. (2) Scaling laws can be noisy and unreliable because they are developed based on pre-training runs conducted with small compute budgets (Wei et al., 2022b).</p>
<p>To address these challenges, we implement a two-stage methodology to develop scaling laws that accurately predict downstream benchmark performance:</p>
<p>1. We first establish a correlation between the compute-optimal model’s negative log-likelihood on downstream tasks and the training FLOPs.</p>
<p>2. Next, we correlate the negative log-likelihood on downstream tasks with task accuracy, utilizing both the scaling law models and older models trained with higher compute FLOPs. In this step, we specifically leverage the Llama 2 family of models.</p>
<p>This approach enables us to predict downstream task performance given a specific number of training FLOPs for compute-optimal models. We use a similar method to select our pre-training data mix (see Section 3.4).</p>
<p>Scaling law experiments. Concretely, we construct our scaling laws by pre-training models using compute budgets between 6 × 10<sup>18</sup> FLOPs and 10<sup>22</sup> FLOPs. At each compute budget, we pre-train models ranging in size between 40M and 16B parameters, using a subset of model sizes at each compute budget. In these training runs, we use a cosine learning rate schedule with a linear warmup for 2,000 training steps. The peak learning rate is set between 2 × 10<sup>-4</sup> and 4 × 10<sup>-4</sup> depending on the size of the model. We set the cosine decay to 0.1 of the peak value. The weight decay at each step is set to 0.1 times the learning rate at that step. We use a fixed batch size for each compute scale, ranging between 250K and 4M.</p>
<p><sup>3</sup>https://github.com/openai/tiktoken/tree/main</p>
<p>7</p>
</html>
3.3 Infrastructure, Scaling, and Efficiency

We describe our hardware and infrastructure that powered Llama 3 405B pre-training at scale and discuss several optimizations that led to improvements in training efficiency.

3.3.1 Training Infrastructure

The Llama 1 and 2 models were trained on Meta’s AI Research SuperCluster (Lee and Sengupta, 2022). As we scaled further, the training for Llama 3 was migrated to Meta’s production clusters (Lee et al., 2024). This
8

Normalized NLL per Char.
1.400
1.0
1.375
0.9
1.350
0.8
1.325
1.300
1.275
Accuracy
0.7
0.6
0.5
1.250
0.4
1.225
1.200
0.3
Scaling Law Models
Llama 2 Models
▲ Scaling Law Prediction
Llama 3 405B
10<sup>20</sup> 10<sup>21</sup> 10<sup>22</sup> 10<sup>23</sup> 10<sup>24</sup> 10<sup>25</sup>
Compute (FLOPs)
1.40
1.30
1.25
1.35
Normalized NLL per Char.
1.20

Figure 4 Scaling law forecast for ARC Challenge. Left: Normalized negative log-likelihood of the correct answer on the ARC Challenge benchmark as a function of pre-training FLOPs. Right: ARC Challenge benchmark accuracy as a function of the normalized negative log-likelihood of the correct answer. This analysis enables us to predict model performance on the ARC Challenge benchmark before pre-training commences. See text for details.

setup optimizes for production-grade reliability, which is essential as we scale up training.

Compute. Llama 3 405B is trained on up to 16K H100 GPUs, each running at 700W TDP with 80GB HBM3, using Meta’s Grand Teton AI server platform (Matt Bowman, 2022). Each server is equipped with eight GPUs and two CPUs. Within a server, the eight GPUs are connected via NVLink. Training jobs are scheduled using MAST (Choudhury et al., 2024), Meta’s global-scale training scheduler.

Storage. Tectonic (Pan et al., 2021), Meta’s general-purpose distributed file system, is used to build a storage fabric (Battey and Gupta, 2024) for Llama 3 pre-training. It offers 240 PB of storage out of 7,500 servers equipped with SSDs, and supports a sustainable throughput of 2 TB/s and a peak throughput of 7 TB/s. A major challenge is supporting the highly bursty checkpoint writes that saturate the storage fabric for short durations. Checkpointing saves each GPU’s model state, ranging from 1 MB to 4 GB per GPU, for recovery and debugging. We aim to minimize GPU pause time during checkpointing and increase checkpoint frequency to reduce the amount of lost work after a recovery.

Network. Llama 3 405B used RDMA over Converged Ethernet (RoCE) fabric based on the Arista 7800 and Minipack2 Open Compute Project<sup>4</sup> OCP rack switches. Smaller models in the Llama 3 family were trained using Nvidia Quantum2 Infiniband fabric. Both RoCE and Infiniband clusters leverage 400 Gbps interconnects between GPUs. Despite the underlying network technology differences between these clusters, we tune both of them to provide equivalent performance for these large training workloads. We elaborate further on our RoCE network since we fully own its design.

• Network topology. Our RoCE-based AI cluster comprises 24K GPUs<sup>5</sup> connected by a three-layer Clos network (Lee et al., 2024). At the bottom layer, each rack hosts 16 GPUs split between two servers and connected by a single Minipack2 top-of-the-rack (ToR) switch. In the middle layer, 192 such racks are connected by Cluster Switches to form a pod of 3,072 GPUs with full bisection bandwidth, ensuring no oversubscription. At the top layer, eight such pods within the same datacenter building are connected via Aggregation Switches to form a cluster of 24K GPUs. However, network connectivity at the aggregation layer does not maintain full bisection bandwidth and instead has an oversubscription ratio of 1:7. Our model parallelism methods (see Section 3.3.2) and training job scheduler (Choudhury et al., 2024) are all optimized to be aware of network topology, aiming to minimize network communication across pods.

• Load balancing. LLM training produces fat network flows that are hard to load balance across all available network paths using traditional methods such as Equal-Cost Multi-Path (ECMP) routing. To address this challenge, we employ two techniques. First, our collective library creates 16 network flows between two GPUs, instead of just one, thereby reducing the traffic per flow and providing more flows

<sup>4</sup>Open Compute Project: https://www.opencompute.org/

<sup>5</sup>Note that we use only up to 16K of these 24K GPUs for Llama 3 pre-training.

9

<html><table><tr><th>GPUs</th><th>TP</th><th>CP</th><th>PP</th><th>DP</th><th>Seq. Len.</th><th>Batch size/DP</th><th>Tokens/Batch</th><th>TFLOPs/GPU</th><th>BF16 MFU</th></tr><tr><td>8,192</td><td>8</td><td>1</td><td>16</td><td>64</td><td>8,192</td><td>32</td><td>16M</td><td>430</td><td>43%</td></tr><tr><td>16,384</td><td>8</td><td>1</td><td>16</td><td>128</td><td>8,192</td><td>16</td><td>16M</td><td>400</td><td>41%</td></tr><tr><td>16,384</td><td>8</td><td>16</td><td>16</td><td>8</td><td>131,072</td><td>16</td><td>16M</td><td>380</td><td>38%</td></tr></table>
<p>Table 4 Scaling configurations and MFU for each stage of Llama 3 405B pre-training. See text and Figure 5 for descriptions of each type of parallelism.</p>
<p>for load balancing. Second, our Enhanced-ECMP (E-ECMP) protocol effectively balances these 16 flows across different network paths by hashing on additional fields in the RoCE header of packets.</p>
<p>•Congestion control. We use deep-buffer switches in the spine (Gangidi et al., 2024) to accommodate transient congestion and buffering caused by collective communication patterns. This setup helps limit the impact of persistent congestion and network back pressure caused by slow servers, which is common in training. Finally, better load balancing through E-ECMP significantly reduces the chance of congestion. With these optimizations, we successfully run a 24K GPU cluster without traditional congestion control methods such as Data Center Quantized Congestion Notification (DCQCN).</p>
<p>3.3.2 Parallelism for Model Scaling</p>
<p>To scale training for our largest models, we use 4D parallelism—a combination of four different types of parallelism methods—to shard the model. This approach efficiently distributes computation across many GPUs and ensures each GPU’s model parameters, optimizer states, gradients, and activations fit in its HBM. Our implementation of 4D parallelism is illustrated in Figure 5. It combines tensor parallelism (TP; Krizhevsky et al. (2012); Shoeybi et al. (2019); Korthikanti et al. (2023)), pipeline parallelism (PP; Huang et al. (2019); Narayanan et al. (2021); Lamy-Poirier (2023)), context parallelism (CP; Liu et al. (2023a)), and data parallelism (DP; Rajbhandari et al. (2020); Ren et al. (2021); Zhao et al. (2023b)).</p>
<p>Tensor parallelism splits individual weight tensors into multiple chunks on different devices. Pipeline parallelism partitions the model vertically into stages by layers, so that different devices can process in parallel different stages of the full model pipeline. Context parallelism divides the input context into segments, reducing memory bottleneck for very long sequence length inputs. We use fully sharded data parallelism (FSDP; Rajbhandari et al. (2020); Ren et al. (2021); Zhao et al. (2023b)), which shards the model, optimizer, and gradients while implementing data parallelism which processes data in parallel on multiple GPUs and synchronizes after each training step. Our use of FSDP for Llama 3 shards optimizer states and gradients, but for model shards we do not reshard after forward computation to avoid an extra all-gather communication during backward passes.</p>
<p>GPU utilization. Through careful tuning of the parallelism configuration, hardware, and software, we achieve an overall BF16 Model FLOPs Utilization (MFU; Chowdhery et al. (2023)) of 38-43% for the configurations shown in Table 4. The slight drop in MFU to 41% on 16K GPUs with DP=128 compared to 43% on 8K GPUs with DP=64 is due to the lower batch size per DP group needed to keep the global tokens per batch constant during training.</p>
<p>Pipeline parallelism improvements. We encountered several challenges with existing implementations:</p>
<p>•Batch size constraint. Current implementations have constraints on supported batch size per GPU, requiring it to be divisible by the number of pipeline stages. For the example in Figure 6, the depth-first schedule (DFS) of pipeline parallelism (Narayanan et al., 2021) requires N=PP= 4, while the breadth-first schedule (BFS; Lamy-Poirier (2023)) requires N=M, where M is the total number of micro-batches and N is the number of contiguous micro-batches for the same stage’s forward or backward. However, pre-training often needs flexibility to adjust batch size.</p>
<p>•Memory imbalance. Existing pipeline parallelism implementations lead to imbalanced resource consumption. The first stage consumes more memory due to the embedding and the warm-up micro-batches.</p>
<p>•Computation imbalance. After the last layer of the model, we need to calculate output and loss, making this stage the execution latency bottleneck.</p>
<p>10</p>


Figure 5 Illustration of 4D parallelism. GPUs are divided into parallelism groups in the order of [TP, CP, PP, DP], where DP stands for FSDP. In this example, 16 GPUs are configured with a group size of |TP|=2, |CP|=2, |PP|=2, and |DP|=2. A GPU’s position in 4D parallelism is represented as a vector, [D<sub>1</sub>, D<sub>2</sub>, D<sub>3</sub>, D<sub>4</sub>], where D<sub>i</sub> is the index on the i-th parallelism dimension. In this example, GPU0[TP0, CP0, PP0, DP0] and GPU1[TP1, CP0, PP0, DP0] are in the same TP group, GPU0 and GPU2 are in the same CP group, GPU0 and GPU4 are in the same PP group, and GPU0 and GPU8 are in the same DP group.

To address these issues, we modify our pipeline schedule as shown in Figure 6, which allows setting N flexibly—in this case N = 5, which can run an arbitrary number of micro-batches in each batch. This allows us to run: (1) fewer micro-batches than the number of stages when we have batch size limit at large scale; or (2) more micro-batches to hide point-to-point communication, finding a sweet spot between DFS and breadth first schedule (BFS) for the best communication and memory efficiency. To balance the pipeline, we reduce one Transformer layer each from the first and the last stages, respectively. This means that the first model chunk on the first stage has only the embedding, and the last model chunk on the last stage has only output projection and loss calculation. To reduce pipeline bubbles, we use an interleaved schedule (Narayanan et al., 2021) with V pipeline stages on one pipeline rank. Overall pipeline bubble ratio is (P-1)/(V*M). Further, we adopt asynchronous point-to-point communication in PP, which considerably speeds up training, especially in cases when the document mask introduces extra computation imbalance. We enable TORCH_NCCL_AVOID_RECORD_STREAMS to reduce memory usage from asynchronous point-to-point communication. Finally, to reduce memory cost, based on detailed memory allocation profiling, we proactively deallocate tensors that will not be used for future computation, including the input and output tensors of each pipeline stage, that will not be used for future computation. With these optimizations, we could pre-train Llama 3 on sequences of 8K tokens without activation checkpointing.

Context parallelism for long sequences. We utilize context parallelism (CP) to improve memory efficiency when scaling the context length of Llama 3 and enable training on extremely long sequences up to 128K in length. In CP, we partition across the sequence dimension, and specifically we partition the input sequence into 2 × CP chunks so each CP rank receives two chunks for better load balancing. The i-th CP rank received both the i-th and the (2 × CP – 1 – i)-th chunks.

Different from existing CP implementations that overlap communication and computation in a ring-like structure (Liu et al., 2023a), our CP implementation adopts an all-gather based method where we first all-gather the key (K) and value (V) tensors, and then compute attention output for the local query (Q) tensor chunk. Although the all-gather communication latency is exposed in the critical path, we still adopt this approach for two main reasons: (1) it is easier and more flexible to support different types of attention masks in all-gather based CP attention, such as the document mask; and (2) the exposed all-gather latency

11

<div>
<img src="llama3_pipeline_parallelism.png" alt="Illustration of pipeline parallelism in Llama 3">
</div>
<p>Figure 6 Illustration of pipeline parallelism in Llama 3. Pipeline parallelism partitions eight pipeline stages (0 to 7) across four pipeline ranks (PP ranks 0 to 3), where the GPUs with rank 0 run stages 0 and 4, the GPUs with P rank 1 run stages 1 and 5, etc. The colored blocks (0 to 9) represent a sequence of micro-batches, where <i>M</i> is the total number of micro-batches and <i>N</i> is the number of continuous micro-batches for the same stage's forward or backward. Our key insight is to make <i>N</i> tunable.</p>
<p>is small as the communicated K and V tensors are much smaller than Q tensor due to the use of GQA (Ainslie et al., 2023). Hence, the time complexity of attention computation is an order of magnitude larger than all-gather (O(S<sup>2</sup>) versus O(S), where S represents the sequence length in the full causal mask), making the all-gather overhead negligible.</p>
<p><b>Network-aware parallelism configuration.</b> The order of parallelism dimensions, [TP, CP, PP, DP], is optimized for network communication. The innermost parallelism requires the highest network bandwidth and lowest latency, and hence is usually constrained to within the same server. The outermost parallelism may spread across a multi-hop network and should tolerate higher network latency. Therefore, based on the requirements for network bandwidth and latency, we place parallelism dimensions in the order of [TP, CP, PP, DP]. DP (i.e., FSDP) is the outermost parallelism because it can tolerate longer network latency by asynchronously prefetching sharded model weights and reducing gradients. Identifying the optimal parallelism configuration with minimal communication overhead while avoiding GPU memory overflow is challenging. We develop a memory consumption estimator and a performance-projection tool which helped us explore various parallelism configurations and project overall training performance and identify memory gaps effectively.</p>
<p><b>Numerical stability.</b> By comparing training loss between different parallelism setups, we fixed several numerical issues that impact training stability. To ensure training convergence, we use FP32 gradient accumulation during backward computation over multiple micro-batches and also reduce-scatter gradients in FP32 across data parallel workers in FSDP. For intermediate tensors, e.g., vision encoder outputs, that are used multiple times in the forward computation, the backward gradients are also accumulated in FP32.</p>
<p><b>3.3.3 Collective Communication</b></p>
<p>Our collective communication library for Llama 3 is based on a fork of Nvidia's NCCL library, called NCCLX. NCCLX significantly improves the performance of NCCL, especially for higher latency networks. Recall that the order of parallelism dimensions is [TP, CP, PP, DP], where DP corresponds to FSDP. The outermost parallelism dimensions, PP and DP, may communicate through a multi-hop network, with latency up to tens of microseconds. The original NCCL collectives—all-gather and reduce-scatter in FSDP, and point-to-point in PP—require data chunking and staged data copy. This approach incurs several inefficiencies, including (1) requiring a large number of small control messages to be exchanged over the network to facilitate data transfer, (2) extra memory-copy operations, and (3) using extra GPU cycles for communication. For Llama 3 training, we address a subset of these inefficiencies by tuning chunking and data transfer to fit our network latencies, which can be as high as tens of microseconds for a large cluster. We also allow small control messages to traverse our network at a higher priority, especially avoiding being head-of-line blocked in deep-buffer core switches. Our ongoing work for future Llama versions involves making deeper changes in NCCLX to holistically address all the aforementioned problems.</p>
<p>12</p>


<table style="width:50%;">
<tr>
<th>Component</th>
<th>Category</th>
<th>Interruption Count</th>
<th>% of Interruptions</th>
</tr>
<tr>
<td>Faulty GPU</td>
<td>GPU</td>
<td>148</td>
<td>30.1%</td>
</tr>
<tr>
<td>GPU HBM3 Memory</td>
<td>GPU</td>
<td>72</td>
<td>17.2%</td>
</tr>
<tr>
<td>Software Bug</td>
<td>Dependency</td>
<td>54</td>
<td>12.9%</td>
</tr>
<tr>
<td>Network Switch/Cable</td>
<td>Network</td>
<td>35</td>
<td>8.4%</td>
</tr>
<tr>
<td>Host Maintenance</td>
<td>Unplanned Maintenance</td>
<td>32</td>
<td>7.6%</td>
</tr>
<tr>
<td>GPU SRAM Memory</td>
<td>GPU</td>
<td>19</td>
<td>4.5%</td>
</tr>
<tr>
<td>GPU System Processor</td>
<td>GPU</td>
<td>17</td>
<td>4.1%</td>
</tr>
<tr>
<td>NIC</td>
<td>Host</td>
<td>7</td>
<td>1.7%</td>
</tr>
<tr>
<td>NCCL Watchdog Timeouts</td>
<td>Unknown</td>
<td>7</td>
<td>1.7%</td>
</tr>
<tr>
<td>Silent Data Corruption</td>
<td>GPU</td>
<td>6</td>
<td>1.4%</td>
</tr>
<tr>
<td>GPU Thermal Interface + Sensor</td>
<td>GPU</td>
<td>6</td>
<td>1.4%</td>
</tr>
<tr>
<td>SSD</td>
<td>Host</td>
<td>3</td>
<td>0.7%</td>
</tr>
<tr>
<td>Power Supply</td>
<td>Host</td>
<td>3</td>
<td>0.7%</td>
</tr>
<tr>
<td>Server Chassis</td>
<td>Host</td>
<td>2</td>
<td>0.5%</td>
</tr>
<tr>
<td>IO Expansion Board</td>
<td>Host</td>
<td>2</td>
<td>0.5%</td>
</tr>
<tr>
<td>Dependency</td>
<td>Dependency</td>
<td>2</td>
<td>0.5%</td>
</tr>
<tr>
<td>CPU</td>
<td>Host</td>
<td>2</td>
<td>0.5%</td>
</tr>
<tr>
<td>System Memory</td>
<td>Host</td>
<td>2</td>
<td>0.5%</td>
</tr>
</table>
<p>Table 5 Root-cause categorization of unexpected interruptions during a 54-day period of Llama 3 405B pre-training. About 78% of unexpected interruptions were attributed to confirmed or suspected hardware issues.</p>
<p>3.3.4 Reliability and Operational Challenges</p>
<p>The complexity and potential failure scenarios of 16K GPU training surpass those of much larger CPU clusters that we have operated. Moreover, the synchronous nature of training makes it less fault-tolerant—a single GPU failure may require a restart of the entire job. Despite these challenges, for Llama 3, we achieved higher than 90% effective training time while supporting automated cluster maintenance, such as firmware and Linux kernel upgrades (Vigraham and Leonhardi, 2024), which resulted in at least one training interruption daily. The effective training time measures the time spent on useful training over the elapsed time.</p>
<p>During a 54-day snapshot period of pre-training, we experienced a total of 466 job interruptions. Of these, 47 were planned interruptions due to automated maintenance operations such as firmware upgrades or operator-initiated operations like configuration or dataset updates. The remaining 419 were unexpected interruptions, which are classified in Table 5. Approximately 78% of the unexpected interruptions are attributed to confirmed hardware issues, such as GPU or host component failures, or suspected hardware-related issues like silent data corruption and unplanned individual host maintenance events. GPU issues are the largest category, accounting for 58.7% of all unexpected issues. Despite the large number of failures, significant manual intervention was required only three times during this period, with the rest of issues handled by automation.</p>
<p>To increase the effective training time, we reduced job startup and checkpointing time, and developed tools for fast diagnosis and problem resolution. We extensively use PyTorch’s built-in NCCL flight recorder (Ansel et al., 2024), a feature that captures collective metadata and stack traces into a ring buffer, and hence allowing us to diagnose hangs and performance issues quickly at scale, particularly with regard to NCCLX. Using this, we efficiently record every communication event and the duration of each collective operation, and also automatically dump tracing data on NCCLX watchdog or heartbeat timeout. We enable more computationally intensive tracing operations and metadata collection selectively as needed live in production through online configuration changes (Tang et al., 2015) without needing a code release or job restart.</p>
<p>Debugging issues in large-scale training is complicated by the mixed use of NVLink and RoCE in our network. Data transfer over NVLink typically occurs through load/store operations issued by CUDA kernels, and failures in either the remote GPU or NVLink connectivity often manifest as stalled load/store operations within CUDA kernels without returning a clear error code. NCCLX enhances the speed and accuracy of failure</p>
<p>13</p>


3.4.3 Annealing

In the final stage of pre-training, we anneal the learning rate to 0 over 100,000 steps. This annealing stage helps to stabilize the model and improve its generalization performance. We found that this annealing schedule was effective in preventing overfitting and improving the model’s performance on downstream tasks. The annealing stage was performed using approximately 100B training tokens.

14

4.1.2 Reward Model Training

To train the reward model, we use pairwise comparisons of model outputs. We collect these comparisons via human annotation, focusing on aspects like helpfulness, harmlessness, and factuality. The reward model is trained to predict which of the two outputs is preferred by humans. This model is then used in the subsequent DPO stage.

4.1.3 Supervised Fine-tuning (SFT)

After training the reward model, we perform supervised fine-tuning (SFT) on the pre-trained language model. This involves training the model on a dataset of human-written demonstrations, aligning its outputs more closely with human preferences. This step helps to improve the model’s overall quality and alignment with human values.

4.1.4 Direct Preference Optimization (DPO)

Finally, we employ Direct Preference Optimization (DPO) to further refine the model. DPO uses the reward model to guide the optimization process, iteratively improving the model’s performance based on human preferences. This iterative process helps to ensure that the final model is well-aligned with human expectations. The DPO process is illustrated in Figure 7.

4.2 Data

Our post-training data consists of several components:

* Human-annotated data: This includes pairwise comparisons of model outputs, used for reward model training.
* Synthetic data: Generated data used to augment the training data and improve model performance.
* High-quality data sources: Upsampled during the annealing phase of pre-training to improve the final model.

4.3 Data Curation Strategies

To enhance the model’s capabilities, we implemented several data curation strategies:

* Reasoning: Improved by carefully selecting and curating data examples that require logical reasoning.
* Coding: Improved by including a diverse set of coding examples and tasks.
* Factuality: Improved by focusing on data sources known for their accuracy and reliability.
* Multilingual: Improved by incorporating data in multiple languages.
* Tool use: Improved by including examples of tool use and integration.
* Long context: Improved by using examples with extended context windows.
* Precise instruction following: Improved by focusing on data that requires precise and nuanced instruction following. These strategies are detailed in Section 4.3.5.


them to different locations (e.g., user, ipython) within a single dialog turn. To support this, we design a new multi-message chat protocol which uses various special header and termination tokens. The header tokens are used to indicate the source and destination of each message in a conversation. Similarly, the termination tokens indicate when it is the time to alternate between human and AI to speak.

4.1.2 Reward Modeling

We train a reward model (RM) covering different capabilities on top of the pre-trained checkpoint. The training objective is the same as Llama 2 except that we remove the margin term in the loss, as we observe diminishing improvements after data scaling. Following Llama 2, we use all of our preference data for reward modeling after filtering out samples with similar responses. In addition to standard preference pair of (chosen, rejected) response, annotations also create a third “edited response” for some prompts, where the chosen response from the pair is further edited for improvement (see Section 4.2.1). Hence, each preference ranking sample has two or three responses with clear ranking (edited > chosen > rejected). We concatenate the prompt and multiple responses into a single row during training with responses randomly shuffled. This is an approximation to the standard scenario of putting the responses in separate rows and computing the scores, but in our ablations, this approach improves training efficiency without a loss in accuracy.

4.1.3 Supervised Finetuning

The reward model is then used to perform rejection sampling on our human annotation prompts, the details of which are described in Section 4.2. Together with this rejection-sampled data and other data sources (including synthetic data), we finetune the pre-trained language model using a standard cross entropy loss on the target tokens (while masking loss on prompt tokens). More details about the data mix can be found in Section 4.2. We refer to this stage as supervised finetuning (SFT; Wei et al., 2022a; Sanh et al., 2022; Wang et al., 2022b), even though many of the training targets are model-generated. Our largest models are finetuned with a learning rate of 10<sup>-5</sup> over the course of 8.5K to 9K steps. We found these hyperparameter settings to work well across different rounds and data mixes.

4.1.4 Direct Preference Optimization

We further train our SFT models with Direct Preference Optimization (DPO; Rafailov et al., 2024) for human preference alignment. For training, we primarily use the most recent batches of preference data collected using the best performing models from the previous alignment rounds. As a result, our training data conforms better to the distribution of the policy model that is being optimized in each round. We also explored on-policy algorithms such as PPO (Schulman et al., 2017), but found that DPO required less compute for large-scale models and performed better, especially on instruction following benchmarks like IFEval (Zhou et al., 2023). For Llama 3, we use a learning rate of 10<sup>-5</sup> and set the β hyper-parameter to be 0.1. In addition, we apply the following algorithmic modifications to DPO:

• Masking out formatting tokens in DPO loss: We mask out special formatting tokens including header and termination tokens (described in Section 4.1.1) from both chosen and rejected responses in the loss to stabilize DPO training. We observe that having these tokens contribute to the loss may lead to undesired model behaviors such as tail repetition or abruptly generating termination tokens. We hypothesize that this is due to the contrastive nature of the DPO loss – the presence of common tokens in both chosen and rejected responses leads to a conflicting learning objective as the model needs to increase and reduce the likelihood of these tokens simultaneously.

• Regularization with NLL loss: We add an additional negative log-likelihood (NLL) loss term with a scaling coefficient of 0.2 on the chosen sequences, similar to Pang et al. (2024). This helps further stabilize DPO training by maintaining desired formatting for generation and preventing the decrease of log probability of chosen responses (Pang et al., 2024; Pal et al., 2024).

4.1.5 Model Averaging

Finally, we average models obtained from experiments using various versions of data or hyperparameters at each RM, SFT, or DPO stage (Izmailov et al., 2019; Wortsman et al., 2022; Li et al., 2022).

16

<table style="width:100%;">
<tr>
<th>Dataset</th>
<th>% of comparisons</th>
<th>Avg. #turns per dialog</th>
<th>Avg. #tokens per example</th>
<th>Avg. #tokens in prompt</th>
<th>Avg. #tokens in response</th>
</tr>
<tr>
<td>General English</td>
<td>81.99%</td>
<td>4.1</td>
<td>1,000.4</td>
<td>36.4</td>
<td>271.2</td>
</tr>
<tr>
<td>Coding</td>
<td>6.93%</td>
<td>3.2</td>
<td>1,621.0</td>
<td>113.8</td>
<td>462.9</td>
</tr>
<tr>
<td>Multilingual</td>
<td>5.19%</td>
<td>1.8</td>
<td>1,299.4</td>
<td>77.1</td>
<td>420.9</td>
</tr>
<tr>
<td>Reasoning and tools</td>
<td>5.89%</td>
<td>1.6</td>
<td>707.7</td>
<td>46.6</td>
<td>129.9</td>
</tr>
<tr>
<td>Total</td>
<td>100%</td>
<td>3.8</td>
<td>1,041.6</td>
<td>44.5</td>
<td>284.0</td>
</tr>
</table>

Table 6 Statistics of human preference data. We list statistics of the internally collected human preference data used for Llama 3 alignment. We ask annotators to perform multi-turn dialogues with the models and make comparisons among responses at each turn. In post-processing, we split each dialogue to multiple examples at a turn level. Each example consists of a prompt (including previous dialog if available) and a response (e.g., chosen or rejected response).

4.1.6 Iterative Rounds

Following Llama 2, we apply the above methods in six rounds. In each cycle, we collect new preference annotations and SFT data, sampling synthetic data from the latest models.

4.2 Post-training Data

The post-training data composition plays a critical role in the usefulness and behavior of language models. In this section, we discuss our human annotation procedures and preference data collection (Section 4.2.1), the composition of our SFT data (Section 4.2.2), and methods for data quality control and cleaning (Section 4.2.3).

4.2.1 Preference Data

Our preference data annotation process is similar to Llama 2. We deploy multiple models for annotation after each round and sample two responses from two different models for each user prompt. These models can be trained with different data mixes and alignment recipes, allowing for different capability strength (e.g., code expertise) and increased data diversity. We ask annotators to rate the strength of their preference by categorizing it into one of four levels, based on how much more they prefer the chosen response over the rejected one: significantly better, better, slightly better, or marginally better. We also incorporate an editing step after preference ranking to encourage annotators to further improve the preferred response. Annotators edit the chosen response directly or prompt the model with feedback to refine its own response. Consequently, a portion of our preference data has three responses ranked (edited>chosen>rejected).

In Table 6, we report the statistics of preference annotations that we use for Llama 3 training. General English covers multiple subcategories such as knowledge-based question and answering or precise instruction-following, which fall outside the scope of specific capabilities. Compared to Llama 2, we observe an increase in the average length of prompt and response, suggesting that we train Llama 3 on more complex tasks. In addition, we implement a quality analysis and human evaluation process to rigorously assess the data collected, allowing us to refine our prompts and provide systematic, actionable feedback to annotators. For example, as Llama 3 improves after each round, we increase prompt complexity accordingly to target areas where the model lags.

In each round of post-training, we use all the preference data that is available at the time for reward modeling, while only using the latest batches from various capabilities for DPO training. For both reward modeling and DPO, we use samples that are labeled as the chosen response being significantly better or better than the rejected counterpart for training and discard samples with similar responses.

4.2.2 SFT Data

Our finetuning data is largely comprised of the following sources:

• Prompts from our human annotation collection with rejection-sampled responses.
• Synthetic data targeting specific capabilities (see Section 4.3 for more details).

17

<table style="width:100%;">
<tr>
<th>Dataset</th>
<th>% of examples</th>
<th>Avg. # turns</th>
<th>Avg. # tokens</th>
<th>Avg. # tokens in context</th>
<th>Avg. # tokens in final response</th>
</tr>
<tr>
<td>General English</td>
<td>52.66%</td>
<td>6.3</td>
<td>974.0</td>
<td>656.7</td>
<td>317.1</td>
</tr>
<tr>
<td>Code</td>
<td>14.89%</td>
<td>2.7</td>
<td>753.3</td>
<td>378.8</td>
<td>374.5</td>
</tr>
<tr>
<td>Multilingual</td>
<td>3.01%</td>
<td>2.7</td>
<td>520.5</td>
<td>230.8</td>
<td>289.7</td>
</tr>
<tr>
<td>Exam-like</td>
<td>8.14%</td>
<td>2.3</td>
<td>297.8</td>
<td>124.4</td>
<td>173.4</td>
</tr>
<tr>
<td>Reasoning and tools</td>
<td>21.19%</td>
<td>3.1</td>
<td>661.6</td>
<td>359.8</td>
<td>301.9</td>
</tr>
<tr>
<td>Long context</td>
<td>0.11%</td>
<td>6.7</td>
<td>38,135.6</td>
<td>37,395.2</td>
<td>740.5</td>
</tr>
<tr>
<td>Total</td>
<td>100%</td>
<td>4.7</td>
<td>846.1</td>
<td>535.7</td>
<td>310.4</td>
</tr>
</table>
<p>Table 7 Statistics of SFT data. We list internally collected SFT data used for Llama 3 alignment. Each SFT example consists of a context (i.e., all conversation turns except the last one) and a final response.</p>
<p>•Small amounts of human-curated data (see Section 4.3 for more details).</p>
<p>As our post-training rounds progress, we develop stronger Llama 3 variants that we use to collect larger datasets that cover a wide range of complex capabilities. In this section, we discuss the details for the rejection-sampling procedure and overall composition of our final SFT datamix.</p>
<p>Rejection sampling. During rejection sampling (RS), for each prompt collected during human annotation (Section 4.2.1) we sample K (typically between 10 and 30) outputs from the latest chat model policy (usually the best performing checkpoint from the previous post-training iteration, or the best performing checkpoint for a particular capability) and use our reward model to select the best candidate, consistent with Bai et al. (2022). In later rounds of post-training, we introduce system prompts to steer RS responses to conform with desirable tone, style, or formatting, which might be different for different capabilities.</p>
<p>To increase the efficiency of rejection sampling, we adopt PagedAttention (Kwon et al., 2023). PagedAttention enhances memory efficiency through dynamic key-value cache allocation. It supports arbitrary output lengths by dynamically scheduling requests based on the current cache capacity. Unfortunately, this carries the risk of swap-out when running out of memory. To eliminate such swap overhead, we define a maximum output length and perform a request only if sufficient memory is available to fit an output with that length. PagedAttention also enables us to share the key-value cache pages for a prompt across all corresponding outputs. Together, this leads to a throughput improvement of over 2× during rejection sampling.</p>
<p>Overall data composition. Table 7 shows data statistics for each broad category of our “helpfulness” mix. While SFT and preference data contain overlapping domains, they are curated differently, yielding distinct count statistics. In Section 4.2.3 we describe techniques for categorizing topic, complexity, and quality of our data samples. In each round of post-training, we adjust our overall data mix carefully across these axes to tune performance across a wide range of benchmarks. Our final data mix epochs multiple times on some high quality sources and downsamples others.</p>
<p>4.2.3 Data Processing and Quality Control</p>
<p>Given that most of our training data is model-generated, it requires careful cleaning and quality control.</p>
<p>Data cleaning. In the early rounds, we observed a number of undesirable patterns common in our data, such as excessive use of emojis or exclamation points. Therefore, we implement a series of rule-based data removal and modification strategies to filter or clean problematic data. For example, to mitigate overly-apologetic tonal issues, we identify overused phrases (such as “I’m sorry” or “I apologize”) and carefully balance the proportion of such samples in our dataset.</p>
<p>Data pruning. We also apply a collection of model-based techniques to remove low-quality training samples and improve overall model performance:</p>
<p>•Topic classification: We first finetune Llama 3 8B into a topic classifier, and perform inference over all data to classify it into both coarsely-grained buckets (“mathematical reasoning”) and fine-grained</p>
<p>18</p>
buckets ("geometry and trigonometry").

• Quality scoring: We use both reward model and Llama-based signals to obtain a quality score for each sample. For an RM-based score, we consider data that is in the top quartile of RM scores as high quality. For a Llama-based score, we prompt Llama 3 checkpoint to rate each sample on a three-point scale for general English data (accuracy, instruction following, and tone/presentation) and a two-point scale for coding data (bug identification and user intention), and consider samples that obtain the maximum score as high quality. The RM and Llama-based scores have high disagreement rates, and we find that combining these signals yield the best recall on our internal test set. Ultimately, we select examples that are marked as high quality by the RM or the Llama-based filter.

• Difficulty scoring: Because we are also interested in prioritizing examples that are more complex for the model, we score data using two measures of difficulty: Instag (Lu et al., 2023) and Llama-based scoring. For Instag, we prompt Llama 3 70B to perform intention tagging of SFT prompts, where more intentions implies more complexity. We also prompt Llama 3 to measure the difficulty (Liu et al., 2024c) of dialogs on a three-point scale.

• Semantic deduplication: Finally, we perform semantic deduplication (Abbas et al., 2023; Liu et al., 2024c). We first cluster complete dialogs using RoBERTa (Liu et al., 2019b) and within each cluster sort them by quality score × difficulty score. We then do greedy selection by iterating through all sorted examples, and only keeping the ones that have maximum cosine similarity less than a threshold to the examples seen so far in the cluster.

4.3 Capabilities

We highlight special efforts to improve performance for specific capabilities such as code (Section 4.3.1), multilinguality (Section 4.3.2), math and reasoning (Section 4.3.3), long context (Section 4.3.4), tool use (Section 4.3.5), factuality (Section 4.3.6), and steerability (Section 4.3.7).

4.3.1 Code

LLMs for code have received significant attention since the release of Copilot and Codex (Chen et al., 2021). Developers are now widely using these models to generate code snippets, debug, automate tasks, and improve code quality. For Llama 3, we target improving and evaluating code generation, documentation, debugging, and review capabilities for the following high priority programming languages: Python, Java, Javascript, C/C++, Typescript, Rust, PHP, HTML/CSS, SQL, bash/shell. Here, we present our work on improving these coding capabilities via training a code expert, generating synthetic data for SFT, improving formatting with system prompt steering, and creating quality filters to remove bad samples from our training data.

Expert training. We train a code expert which we use to collect high quality human annotations for code throughout subsequent rounds of post-training. This is accomplished by branching the main pre-training run and continuing pre-training on a 1T token mix of mostly (>85%) code data. Continued pre-training on domain-specific data has been shown to be effective for improving performance in a specific domain (Gururangan et al., 2020). We follow a recipe similar to that of CodeLlama (Rozière et al., 2023). For the last several thousand steps of training we perform long-context finetuning (LCFT) to extend the expert’s context length to 16K tokens on a high quality mix of repo-level code data. Finally, we follow the similar post-training modeling recipes described in Section 4.1 to align this model, except with SFT and DPO data mixes primarily targeting code. This model is also used for rejection sampling (Section 4.2.2) for coding prompts.

Synthetic data generation. During development, we identified key issues in code generation, including difficulty in following instructions, code syntax errors, incorrect code generation, and difficulty in fixing bugs. While intensive human annotation could theoretically resolve these issues, synthetic data generation offers a complementary approach at a lower cost and higher scale, unconstrained by the expertise level of annotators. As such, we use Llama 3 and the code expert to generate a large quantity of synthetic SFT dialogs.

We describe three high-level approaches for generating synthetic code data. In total, we generate over 2.7M synthetic examples which were used during SFT.

19

1. Synthetic data generation: execution feedback. The 8B and 70B models show significant performance improvements when trained on data generated by a larger, more competent model. However, our initial experiments revealed that training Llama 3 405B on its own generated data is not helpful (and can even degrade performance). To address this limitation, we introduced execution feedback as a source of truth, enabling the model to learn from its mistakes and stay on track. In particular, we generate a large dataset of approximately one million synthetic coding dialogues using the following process:

• Problem description generation: First, we generate a large collection of programming problem descriptions that span a diverse range of topics, including those in the long tail distribution. To achieve this diversity, we sample random code snippets from various sources and prompt the model to generate programming problems inspired by these examples. This allowed us to tap into a wide range of topics and create a comprehensive set of problem descriptions (Wei et al., 2024).

• Solution generation: Then, we prompt Llama 3 to solve each problem in a given programming language. We observe that adding general rules of good programming to the prompt improves the generated solution quality. Also, we find it is helpful to require the model to explain its thought process in comments.

• Correctness analysis: After generating a solution, it is crucial to recognize that its correctness is not guaranteed, and including incorrect solutions in the finetuning dataset could harm the model's quality. While we do not ensure complete correctness, we develop methods to approximate it. To achieve this, we extract the source code from the generated solution and applied a combination of static and dynamic analysis techniques to test its correctness, including:

– Static analysis: We run all generated code through a parser and a linter to ensure syntactic correctness, catching errors such as syntax errors, use of uninitialized variables or non-imported functions, code style issues, typing errors, and others.

– Unit test generation and execution: For each problem and solution, we prompt the model to generate unit tests, executed in a containerized environment together with the solution, catching run-time execution errors and some semantic errors.

• Error feedback and iterative self-correction: When a solution fails at any step, we prompt the model to revise it. The prompt included the original problem description, the faulty solution, and feedback from the parser/linter/tester (stdout, stderr/ and return code). After a unit test execution failure, the model could either fix the code to pass the existing tests or modify its unit tests to accommodate the generated code. Only dialogs that pass all checks are included in the final dataset, used for supervised finetuning (SFT). Notably, we observed that about 20% of solutions were initially incorrect but self-corrected, indicating that the model learned from the execution feedback and improved its performance.

• Fine-tuning and iterative improvement: The finetuning process is conducted over multiple rounds, with each round building on the previous one. After each round, the model is improved, generating higher-quality synthetic data for the next round. This iterative process allows for progressive refinement and enhancement of the model's performance.

2. Synthetic data generation: programming language translation. We observe a performance gap between major programming languages (e.g., Python/C++) and less common ones (e.g., Typescript/PHP). This is not surprising as we have less training data for less common programming languages. To mitigate this, we supplement our existing data by translating data from common programming languages to less common languages (similar to Chen et al. (2023) in the context of reasoning). This is achieved by prompting Llama 3 and ensuring quality via syntax parsing, compilation, and execution. Figure 8 demonstrates an example of synthetic PHP code translated from Python. This improves performance significantly for less common languages as measured by the MultiPL-E (Cassano et al., 2023) benchmark.

3. Synthetic data generation: backtranslation. To improve certain coding capabilities (e.g., documentation, explanations) where execution feedback is less informative for determining quality, we employ an alternative multi-step approach. Using this procedure, we generated approximately 1.2M synthetic

20

Figure 8 Code translation example. We display an example of using Llama 3 to translate Python code (left) to PHP code (right) to augment our SFT dataset with a wider range of programming languages.

Figure 9 Improving generated code quality with system prompts. Left: without system prompt Right: with system prompt.

dialogs related to code explanation, generation, documentation, and debugging. Beginning with code snippets from a variety of languages in our pre-training data:

• Generate: We prompt Llama 3 to generate data that represents our target capability (e.g., we add comments and docstrings for the code snippet, or we ask the model to explain a piece of code).

• Backtranslate: We then prompt the model to “backtranslate” the synthetically generated data to the original code (e.g., we prompt the model to generate code only from its documentation, or we ask the model to generate code only from its explanation).

• Filter: Using the original code as a reference, we prompt the Llama 3 to determine the quality of the output (e.g., we ask the model how faithful the backtranslated code is to the original). We then use the generated examples that have the highest self-verification scores in SFT.

System prompt steering during rejection sampling. During the rejection sampling process, we used code-specific system prompts to improve code readability, documentation, thoroughness, and specificity. Recall, from Section 7, this data is used to finetune the language model. Figure 9 shows an example of how the system prompt helps improve the generated code quality — it adds necessary comments, uses more informative variable names, saves memory, etc.

Filtering training data with execution and model-as-judge signals. As described in Section 4.2.3, we occasionally encounter quality issues in our rejection-sampled data, such as code blocks containing bugs. Detecting these issues in our rejection-sampled data is not as straightforward as it is for our synthetic code data, as the rejection-sampled responses typically contain a mix of natural language and code for which the code may not

21

4.3.3 Quantitative Reasoning

To further improve Llama 3’s quantitative reasoning capabilities, we leverage a large-scale dataset of synthetic quantitative reasoning problems. This dataset consists of approximately 1.2M examples, each comprising a problem description, a solution, and a rationale. The problems cover a wide range of topics, including arithmetic, algebra, geometry, and probability. The problems are designed to be challenging but solvable, and the solutions are carefully crafted to be both correct and easy to understand. We use this dataset to fine-tune Llama 3, resulting in significant improvements in its quantitative reasoning abilities. We also translate this dataset into several non-English languages to improve performance in those languages. The translation process is described in Section 4.3.2. We evaluate the performance of Llama 3 on several benchmark datasets, including GSM8K and MATH. The results show that Llama 3 achieves state-of-the-art performance on these datasets.

23

correct the incorrect reasoning traces. This allows the model to learn from its mistakes and improve its reasoning abilities. We also incorporate curriculum learning, starting with simpler problems and gradually increasing the difficulty. This helps the model to learn progressively and avoid being overwhelmed by complex problems early on. Finally, we use reinforcement learning to further improve the model's reasoning capabilities. We design a reward function that encourages the model to generate correct answers and step-by-step reasoning traces. This reward function is used to train the model to produce high-quality reasoning traces.

4.3.4 Long Context

During the final pre-training stage, we extend the context length of Llama 3 from 8K tokens to 128K tokens (see Section 3.4 for more details). Similar to pre-training, we find that during finetuning we must carefully tune the recipe to balance short and long-context capabilities.

SFT and synthetic data generation. Naively applying our existing SFT recipe with only short-context data resulted in significant regressions in long-context capabilities from pre-training, highlighting the need to incorporate long-context data in our SFT data mix. In practice, however, it is largely impractical to get humans to annotate such examples due to the tedious and time-consuming nature of reading lengthy contexts, so we predominantly rely on synthetic data to fill this gap. We use earlier versions of Llama 3 to generate synthetic data based on the key long-context use-cases: (possibly multi-turn) question-answering, summarization for long documents, and reasoning over code repositories, and describe them in greater detail below.

• Question answering: We carefully curate a set of long documents from our pre-training mix. We split these documents into chunks of 8K tokens, and prompted an earlier version of the Llama 3 model to generate QA pairs conditional on randomly selected chunks. During training, the whole document is used as context.

• Summarization: We applied hierarchical summarization of long-context documents by first summarizing the chunks of 8K input length using our strongest Llama 3 8K context model and then summarizing the summaries. During training we provide the full document and prompt the model to summarize the document while preserving all the important details. We also generate QA pairs based on the summaries of the documents and prompt the model with questions that require global understanding of the whole long document.

• Long context code reasoning: We parse Python files to identify import statements and determine their dependencies. From here, we select the most commonly depended-upon files, specifically those referenced by at least five other files. We remove one of these key files from a repository and prompt the model to identify which files depended on the missing file and to generate the necessary missing code.

We further categorize these synthetically generated samples based on the sequence length (16K, 32K, 64K and 128K) to enable more fine-grained targeting of input lengths.

Through careful ablations, we observe that mixing 0.1% of synthetically generated long-context data with the original short-context data optimizes the performance across both short-context and long-context benchmarks.

DPO. We observe that using only short context training data in DPO did not negatively impact long-context performance as long as the SFT model is high quality in long context tasks. We suspect this is due to the fact that our DPO recipe has fewer optimizer steps than SFT. Given this finding, we keep the standard short-context recipe for DPO on top of our long-context SFT checkpoints.

4.3.5 Tool Use

Teaching LLMs to use tools such as search engines or code interpreters hugely expands the range of tasks they can solve, transforming them from pure chat models into more general assistants (Nakano et al., 2021; Thoppilan et al., 2022; Parisi et al., 2022; Gao et al., 2023; Mialon et al., 2023a; Schick et al., 2024). We train Llama 3 to interact with the following tools:

• Search engine. Llama 3 is trained to use Brave Search<sup>7</sup> to answer questions about recent events that go beyond its knowledge cutoff or that require retrieving a particular piece of information from the web.

• Python interpreter. Llama 3 can generate and execute code to perform complex computations, read files uploaded by the user and solve tasks based on them such as question answering, summarization, data analysis or visualization.

<sup>7</sup>https://brave.com/search/api/

24

• Mathematical computational engine. Llama 3 can use the Wolfram Alpha API<sup>8</sup> to more accurately solve math, science problems, or retrieve accurate information from Wolfram’s database. The resulting model is able to use these tools in a chat setup to solve the user’s queries, including in multi-turn dialogs. If a query requires multiple tool calls, the model can write a step-by-step plan, call the tools in sequence, and do reasoning after each tool call. We also improve Llama 3’s zero-shot tool use capabilities — given in-context, potentially unseen tool definitions and a user query, we train the model to generate the correct tool call.

Implementation. We implement our core tools as Python objects with different methods. Zero-shot tools can be implemented as Python functions with descriptions, documentation (i.e., examples for how to use them), and the model only needs the function’s signature and docstring as context to generate the appropriate call. We also convert function definitions and calls to JSON format, e.g., for web API calls. All tool calls are executed by the Python interpreter, that must be enabled in the Llama 3 system prompt. Core tools can be individually enabled or disabled in the system prompt.

Data collection. Different from Schick et al. (2024), we rely on human annotations and preferences to teach Llama 3 to use tools. There are two main differences with the post-training pipeline generally used in Llama 3:
• For tools, dialogs often contain more than a single assistant message (e.g., calling the tool and reasoning about the tool output). Thus, we annotate at the message level to collect granular feedback: annotators provide a preference between two assistant messages with the same context or, if both contain major problems, edit one of the messages. The chosen or edited message is then added to the context and the dialog continues. This provides human feedback for both the assistant’s ability of calling the tools and reasoning about the tool outputs. Annotators cannot rank or edit the tool outputs.
• We do not perform rejection sampling, as we did not observe gains in our tool benchmarks.

To accelerate the annotation process, we start by bootstrapping basic tool use capabilities by finetuning on synthetically generated data from previous Llama 3 checkpoints. Thus, annotators have fewer edits to perform. In a similar spirit, as Llama 3 gradually improves through its development, we progressively complexify our human annotation protocols: we start by single-turn tool use annotations, before moving to tool use in dialogs, and finally annotating for multi-step tool use and data analysis.

Tool datasets. To create data for tool usage applications, we leverage the following procedure:
• Single-step tool use: We start by few-shot generation of synthetic user prompts which, by construction, require a call to one of our core tools (for example, questions that exceed our knowledge cutoff date). Then, still relying on few-shot generation, we generate appropriate tool calls for these prompts, execute them, and add the output to the model’s context. Finally, we prompt the model again to generate a final answer to the user’s query based on the tool output. We end up with trajectories of the following form: system prompt, user prompt, tool call, tool output, final answer. We also filter around 30% this dataset to remove tool calls that cannot be executed or other formatting issues.
• Multi-step tool use: We follow a similar protocol and first generate synthetic data to teach the model basic multi-step tool use capabilities. To do this, we first prompt Llama 3 to generate user prompts that require at least two tool calls, that can be the same or different tools from our core set. Then, conditioned on these prompts, we few-shot prompt Llama 3 to generate a solution consisting of interleaved reasoning steps and tool calls, similar to ReAct (Yao et al., 2022). See Figure 10 for an example of Llama 3 performing a task involving multi-step tool usage.
• File uploads: We annotate for the following filetypes: .txt, .docx, .pdf, .pptx, .xlsx, .csv, .tsv, .py, .json, .jsonl, .html, .xml. Our prompts are based on a provided file, and ask to summarize the contents of the file, find and fix bugs, optimize a piece of code, perform data analysis or visualization. See Figure 11 for an example of Llama 3 performing a task involving a file upload.

After finetuning on this synthetic data, we gather human annotations in diverse and challenging scenarios including multi-turn interactions, more than three step tool use, and instances where a tool call does not yield
<sup>8</sup>https://products.wolframalpha.com/llm-api/documentation
25

Figure 10 Multi-step tool usage. Example of Llama 3 performing multi-step planning, reasoning, and tool calling to solve a task.

a satisfying answer. We augment our synthetic data with different system prompts to teach the model to use tools only when activated. To train the model to avoid calling tools for simple queries, we also add queries from easy math or question answering datasets (Berant et al., 2013; Koncel-Kedziorski et al., 2016; Joshi et al., 2017; Amini et al., 2019) and their responses without tools, but with tools activated in system prompt.

Zero-shot tool use data. We improve Llama 3 zero-shot tool use abilities (also referred to as function calling) by finetuning on a large and diverse set of partly synthetic (functions definitions, user query, corresponding call) tuples. We evaluate our model on a set of unseen tools.

•	Single, nested, and parallel function calling: Calls can be simple, nested, i.e. we pass a function call as an argument of another function, or parallel, i.e. the model returns a list of independent function calls. Generating a diverse set of functions, queries and ground truths can be challenging (Mekala et al., 2024), and we resort to mining the Stack (Kocetkov et al., 2022) to ground our synthetic user queries in real functions. More precisely, we extract function calls and their definitions, clean and filter them, e.g. for missing docstrings or non-executable functions, and use Llama 3 to generate a natural language query corresponding to the function call.

•	Multi-turn function calling: We also generate synthetic data for multi-turn dialogs with function calls, following a protocol similar to the one proposed in Li et al. (2023b). We use multiple agents that generate domains, APIs, user queries, API calls, and responses, while also ensuring that the generated data covers a set of diverse domains and realistic APIs. All agents are variants of Llama 3 prompted in different ways depending on their roles and collaborate in a step-by-step manner.

4.3.6 Factuality

Hallucinations remain a major challenge for large language models. Models tend to be overconfident, even in domains where they have little knowledge. Despite these shortcomings, they are often used as knowledge bases, which can lead to risky outcomes such as the spread of misinformation. While we recognize that factuality can go beyond hallucinations, we took a hallucination-first approach here.

26

Figure 11 Processing file uploads. Example of Llama 3 performing analysis and visualization of an uploaded file.

We follow the principle that post-training should align the model to “know what it knows” rather than add knowledge (Gekhman et al., 2024; Mielke et al., 2020). Our primary approach involves generating data that aligns model generations with subsets of factual data present in the pre-training data. To achieve this, we develop a knowledge probing technique that takes advantage of Llama 3’s in-context abilities. This data generation process involves the following procedure:

1. Extract a data snippet from the pre-training data.
2. Generate a factual question about these snippets (context) by prompting Llama 3.
3. Sample responses from Llama 3 to the question.
4. Score the correctness of the generations using the original context as a reference and Llama 3 as a judge.
5. Score the informativeness of the generations using Llama 3 as a judge.
6. Generate a refusal for responses which are consistently informative and incorrect across the generations, using Llama 3.

We use data generated from the knowledge probe to encourage the model to only answer questions which it has knowledge about, and refuse answering those questions that it is unsure about. Further, pre-training data is not always factually consistent or correct. We therefore also collect a limited set of labeled factuality data that deals with sensitive topics where factually contradictory or incorrect statements are prevalent.

27

5.1.1 Standard Benchmarks

To compare our models with the current state-of-the-art, we evaluate Llama 3 on a large number of standard benchmark evaluations shown in Table 8. These evaluations cover eight top-level categories: (1) commonsense reasoning; (2) knowledge; (3) reading comprehension; (4) math, reasoning, and problem solving; (5) long context; (6) code; (7) adversarial evaluations; and (8) aggregate evaluations.

28

Reading Comprehension | SQUAD V2 (Rajpurkar et al., 2018), QuaC (Choi et al., 2018), RACE (Lai et al., 2017),
---|---|
Code | HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021),
---|---|
Commonsense reasoning/understanding | CommonSenseQA (Talmor et al., 2019), PiQA (Bisk et al., 2020), SiQA (Sap et al., 2019), OpenBookQA (Mihaylov et al., 2018), WinoGrande (Sakaguchi et al., 2021)
---|---|
Math, reasoning, and problem solving | GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021b), ARC Challenge (Clark et al., 2018), DROP (Dua et al., 2019), WorldSense (Benchekroun et al., 2023)
---|---|
Adversarial | Adv SQUAD (Jia and Liang, 2017), Dynabench SQUAD (Kiela et al., 2021), GSM-Plus (Li et al., 2024c) PAWS (Zhang et al., 2019)
---|---|
Long context | QuALITY (Pang et al., 2022), many-shot GSM8K (An et al., 2023a)
---|---|
Aggregate | MMLU (Hendrycks et al., 2021a), MMLU-Pro (Wang et al., 2024b), AGIEval (Zhong et al., 2023), BIG-Bench Hard (Suzgun et al., 2023)
---|---|

Table 8 Pre-training benchmarks by category. Overview of all benchmarks we use to evaluate pre-trained Llama 3 models, grouped by capability category.

Experimental setup. For each benchmark, we compute scores for Llama 3 as well as various other pre-trained models of comparable sizes. Where possible, we recompute numbers with our own pipeline for other models. To ensure a fair comparison, we then select the best score between the score that we computed and the reported number for that model with comparable or more conservative settings. You can find additional details on our evaluation setup here. For some models, it is not possible to (re)compute benchmark values, for instance, because the pre-trained model is not released or because the API does not provide access to log-probabilities. In particular, this is true for all models comparable to Llama 3 405B. Thus, we do not report category averages for Llama 3 405B, which requires that all numbers are available for all benchmarks.

Significance estimates. Benchmark scores are estimates of a model’s true performance. These estimates have variance because benchmark sets are finite samples drawn from some underlying distribution. We follow Madaan et al. (2024b) and report on this variance via 95% confidence intervals (CIs), assuming that benchmark scores are Gaussian distributed. While this assumption is incorrect (e.g., benchmark scores are bounded), preliminary bootstrap experiments suggest CIs (for discrete metrics) are a good approximation:

CI(S) = 1.96 × √(S × (1 - S)) / N.

Herein, S is the observed benchmark score (e.g., accuracy or EM) and N the sample size of the benchmark. We omit CIs for benchmark scores that are not simple averages. We note that because subsampling is not the only source of variation, our CI values lower bound the actual variation in the capability estimate.

Results for 8B and 70B models. Figure 12 reports the average performance of Llama 3 8B and 70B on the commonsense reasoning, knowledge, reading comprehension, math and reasoning, and code benchmarks. The results show that Llama 3 8B outperforms competing models in virtually every category, both in terms of per-category win rate and in terms of average per-category performance. We also find that Llama 3 70B outperforms its predecessor Llama 2 70B by a large margin on most benchmarks, with the exception of commonsense benchmarks that are likely saturated. Llama 3 70B also outperforms Mixtral 8x22B.

Detailed results for all models. Table 9, 10, 11, 12, 13, and 14 present the benchmark performance of pre-trained Llama 3 8B, 70B, and 405B models on reading comprehension tasks, coding tasks, commonsense understanding tasks, mathematical reasoning tasks, and general tasks. The tables compare Llama 3’s performance with that

29

5.1.2 Model Robustness

In addition to performance on benchmarks, robustness is an important factor in the quality of pre-trained language models. We investigate the robustness of our pre-trained language models to design choices in multiple-choice question (MCQ) setups. Prior work has reported that model performance can be sensitive to seemingly arbitrary design choices in such setups, for example, model scores and even rankings may change with the order and labels of the in-context examples (Lu et al., 2022; Zhao et al., 2021; Robinson and Wingate, 2023; Liang et al., 2022; Gupta et al., 2024), the exact format of the prompt (Weber et al., 2023b; Mishra et al., 2022), or the answer choice format and order (Alzahrani et al., 2024; Wang et al., 2024a; Zheng et al., 2023). Motivated by this work, we use the MMLU benchmark to evaluate the robustness of our pre-trained models to: (1) few-shot label bias, (2) label variants, (3) answer order, and (4) prompt format:

• Few-shot label bias. Following Zheng et al. (2023) and Weber et al. (2023a), we investigate the impact of the distribution of labels in four-shot examples. Specifically, we consider settings in which: (1) all

30

Commonsense Understanding
CommonSenseQA	PiQA	SiQA	OpenBookQA	Winogrande
Llama 3 8B	75.0 ±2.5	81.0 ±1.8	49.5 ±2.2	45.0 ±4.4	75.7 ±2.0
Mistral 7B	71.2 ±2.6	83.0 ±1.7	48.2 ±2.2	47.8 ±4.4	78.1 ±1.9
Gemma 7B	74.4 ±2.5	81.5 ±1.8	51.8 ±2.2	52.8 ±4.4	74.7 ±2.0
Llama 3 70B	84.1 ±2.1	83.8 ±1.7	52.2 ±2.2	47.6 ±4.4	83.5 ±1.7
Mixtral 8×22B	82.4 ±2.2	85.5 ±1.6	51.6 ±2.2	50.8 ±4.4	84.7 ±1.7
Llama 3 405B	85.8 ±2.0	85.6 ±1.6	53.7 ±2.2	49.2 ±4.4	82.2 ±1.8
GPT-4	–	–	–	–	87.5 ±1.5
Nemotron 4 340B	–	–	–	–	89.5 ±1.4
Table 11 Pre-trained model performance on commonsense understanding tasks. Results include 95% confidence intervals.

Math and Reasoning
GSM8K	MATH	ARC-C	DROP	WorldSense
Llama 3 8B	57.2 ±2.7	20.3 ±1.1	79.7 ±2.3	59.5 ±1.0	45.5 ±0.3
Mistral 7B	52.5 ±2.7	13.1 ±0.9	78.2 ±2.4	53.0 ±1.0	44.9 ±0.3
Gemma 7B	46.4 ±2.7	24.3 ±1.2	78.6 ±2.4	56.3 ±1.0	46.0 ±0.3
Llama 3 70B	83.7 ±2.0	41.4 ±1.4	92.9 ±1.5	79.6 ±0.8	61.1 ±0.3
Mixtral 8×22B	88.4 ±1.7	41.8 ±1.4	91.9 ±1.6	77.5 ±0.8	51.5 ±0.3
Llama 3 405B	89.0 ±1.7	53.8 ±1.4	96.1 ±1.1	84.8 ±0.7	63.7 ±0.3
GPT-4	92.0 ±1.5	–	96.3 ±1.1	80.9 ±0.8	–
Nemotron 4 340B	–	–	94.3 ±1.3	–	–
Gemini Ultra	88.9<sup>♢</sup>±1.7	53.2 ±1.4	–	82.4<sup>△</sup>±0.8	–
Table 12 Pre-trained model performance on math and reasoning tasks. Results include 95% confidence intervals. <sup>♢</sup>11-shot. <sup>△</sup>Variable shot.

General
MMLU	MMLU-Pro	AGIEval	BB Hard
Llama 3 8B	66.7	37.1	47.8 ±1.9	64.2 ±1.2
Mistral 7B	63.6	32.5	42.7 ±1.9	56.8 ±1.2
Gemma 7B	64.3	35.1	46.0 ±1.9	57.7 ±1.2
Llama 3 70B	79.3	53.8	64.6 ±1.9	81.6 ±0.9
Mixtral 8×22B	77.8	51.5	61.5 ±1.9	79.5 ±1.0
Llama 3 405B	85.2	61.6	71.6 ±1.8	85.9 ±0.8
GPT-4	86.4	–	–	–
Nemotron 4 340B	81.1	–	–	85.4 ±0.9
Gemini Ultra	83.7	–	–	83.6 ±0.9
Table 13 Pre-trained model performance on general language tasks. Results include 95% confidence intervals.
31

Micro accuracy

<table>
  <tr>
    <td></td>
    <td>Llama 3 8B</td>
    <td>Llama 3 70B</td>
    <td>Llama 3 405B</td>
  </tr>
  <tr>
    <td>[A. B. C. D.]</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>[A) B) C) D)]</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>[1 2 3 4]</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>[$ & # @]</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>[œ § 3 ü]</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

<br>

Micro accuracy

<table>
  <tr>
    <td></td>
    <td>Llama 3 8B</td>
    <td>Llama 3 70B</td>
    <td>Llama 3 405B</td>
  </tr>
  <tr>
    <td>ABCD</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>AADD</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>BBCC</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>AAAA</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

<br>

Figure 13 Robustness of our pre-trained language models to different design choices in the MMLU benchmark. Left: Performance for different label variants. Right: Performance for different labels present in few-shot examples.

<br>

Micro accuracy

<table>
  <tr>
    <td></td>
    <td>Llama 3 8B</td>
    <td>Llama 3 70B</td>
    <td>Llama 3 405B</td>
  </tr>
  <tr>
    <td>Permutation distance 0</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>2</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>3</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>4</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

<br>

Micro accuracy

<table>
  <tr>
    <td></td>
    <td>Llama 3 8B</td>
    <td>Llama 3 70B</td>
    <td>Llama 3 405B</td>
  </tr>
</table>

<br>

Figure 14 Robustness of our pre-trained language models to different design choices in the MMLU benchmark. Left: Performance for different answer orders. Right: Performance for different prompt formats.

few-shot examples have the same label (AAAA); (2) all examples have a different label (ABCD); and (3) there are only two labels present (AABB and CCDD).

• Label variants. We also study model response to different choice token sets. We consider the two sets proposed by Alzahrani et al. (2024): namely, a set of common language independent tokens ($ & # @) and a set of rare tokens (œ § зü) that do not have any implicit relative order. We also consider two versions of the canonical labels (A. B. C. D. and A) B) C) D)) and a numerical list (1, 2, 3, 4).

• Answer order. Following Wang et al. (2024a), we compute how stable the results are across different answer orders. To compute this, we remap all the answers in the dataset according to a fixed permutation. For example, for the permutation ABCD, all answer options with label A and B keep their label, and all answer options with label C get label D, and vice versa.

• Prompt format. We evaluate variance in performance across five task prompts that differ in the level of information provided: one prompt simply asks the model to answer the question, whereas other prompts assert the expertise of the model or that the best answer should be chosen.

Figure 13 presents the results of our experiments studying robustness of model performance to label variants (left) and few-shot label bias (right). The results show that our pre-trained language models are very robust to changes in MCQ labels and to the structure of the few-shot prompt labels. This robustness is particularly

32

5.1.4 Contamination Analysis

We conduct a contamination analysis to estimate to what extent benchmark scores may be influenced by contamination of the evaluation data in the pre-training corpus. In previous work, several different contamination methods have been used, with various different hyperparameters — we refer to Singh et al. (2024) for an overview. Any of these methods can suffer from false positives and negatives, and how to best run contamination analyses is currently still an open field of research. Here, we largely follow the suggestions of Singh et al. (2024).

33

<div>
<p>Llama 3</p>
<table>
<tr>
<th></th>
<th>8B</th>
<th>70B</th>
<th>405B</th>
</tr>
<tr>
<td>QuALITY (5-shot)</td>
<td>56.0 ±2.1</td>
<td>82.8 ±1.6</td>
<td>87.6 ±1.4</td>
</tr>
<tr>
<td>GSM8K (16-shot)</td>
<td>60.0 ±9.6</td>
<td>83.0 ±7.4</td>
<td>90.0 ±5.9</td>
</tr>
</table>
<p>Table 14 Performance of pre-trained models on long-context tasks. Results include 95% confidence intervals.</p>
<table>
<tr>
<th></th>
<th>Contam.</th>
<th>Performance gain est.</th>
</tr>
<tr>
<th></th>
<th>8B</th>
<th>70B</th>
<th>405B</th>
</tr>
<tr>
<td>AGIEval</td>
<td>98</td>
<td>8.5</td>
<td>19.9</td>
<td>16.3</td>
</tr>
<tr>
<td>BIG-Bench Hard</td>
<td>95</td>
<td>26.0</td>
<td>36.0</td>
<td>41.0</td>
</tr>
<tr>
<td>BoolQ</td>
<td>96</td>
<td>4.0</td>
<td>4.7</td>
<td>3.9</td>
</tr>
<tr>
<td>Common SenseQA</td>
<td>30</td>
<td>0.1</td>
<td>0.8</td>
<td>0.6</td>
</tr>
<tr>
<td>DROP</td>
<td>–</td>
<td>–</td>
<td>–</td>
<td>–</td>
</tr>
<tr>
<td>GSM8K</td>
<td>41</td>
<td>0.0</td>
<td>0.1</td>
<td>1.3</td>
</tr>
<tr>
<td>HellaSwag</td>
<td>85</td>
<td>14.8</td>
<td>14.8</td>
<td>14.3</td>
</tr>
<tr>
<td>HumanEval</td>
<td>–</td>
<td>–</td>
<td>–</td>
<td>–</td>
</tr>
<tr>
<td>MATH</td>
<td>1</td>
<td>0.0</td>
<td>-0.1</td>
<td>-0.2</td>
</tr>
<tr>
<td>MBPP</td>
<td>–</td>
<td>–</td>
<td>–</td>
<td>–</td>
</tr>
<tr>
<td>MMLU</td>
<td>–</td>
<td>–</td>
<td>–</td>
<td>–</td>
</tr>
<tr>
<td>MMLU-Pro</td>
<td>–</td>
<td>–</td>
<td>–</td>
<td>–</td>
</tr>
<tr>
<td>NaturalQuestions</td>
<td>52</td>
<td>1.6</td>
<td>0.9</td>
<td>0.8</td>
</tr>
<tr>
<td>OpenBookQA</td>
<td>21</td>
<td>3.0</td>
<td>3.3</td>
<td>2.6</td>
</tr>
<tr>
<td>PiQA</td>
<td>55</td>
<td>8.5</td>
<td>7.9</td>
<td>8.1</td>
</tr>
<tr>
<td>Quac</td>
<td>99</td>
<td>2.4</td>
<td>11.0</td>
<td>6.4</td>
</tr>
<tr>
<td>RACE</td>
<td>–</td>
<td>–</td>
<td>–</td>
<td>–</td>
</tr>
<tr>
<td>SiQA</td>
<td>63</td>
<td>2.0</td>
<td>2.3</td>
<td>2.6</td>
</tr>
<tr>
<td>SQuAD</td>
<td>0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
</tr>
<tr>
<td>Winogrande</td>
<td>6</td>
<td>-0.1</td>
<td>-0.1</td>
<td>-0.2</td>
</tr>
<tr>
<td>WorldSense</td>
<td>73</td>
<td>-3.1</td>
<td>-0.4</td>
<td>3.9</td>
</tr>
</table>
<p>Table 15 Percentage of evaluation sets considered to be contaminated because similar data exists in the training corpus, and the estimated performance gain that may result from that contamination. See the text for details.</p>
<p>Method. Specifically, Singh et al. (2024) propose to select contamination detection methods empirically, based on which method results in the largest difference between the ‘clean’ part of the dataset and the entire dataset, which they call estimated performance gain. For all our evaluation datasets, we score examples based on 8-gram overlap, a method that was found by Singh et al. (2024) to be accurate for many datasets. We consider an example of a dataset D to be contaminated if a ratio TD of its tokens are part of an 8-gram occurring at least once in the pre-training corpus. We select TD separately for each dataset, based on which value shows the maximal significant estimated performance gain across the three model sizes.</p>
<p>Results. In Table 15, we report the percentage of evaluation data that is considered contaminated for the maximal estimated performance gain, as described above, for all key benchmarks. From the table, we exclude numbers for benchmarks for which the results are not significant, for instance because the clean or contaminated set has too few examples, or because the observed performance gain estimate shows extremely erratic behavior. In Table 15, we observe that for some datasets contamination has a large impact, while for others it does not. For example, for PiQA and HellaSwag, both the estimation of contamination and the estimation of performance gain are high. For Natural Questions, on the other hand, the estimated 52% contamination seems to have virtually no effect on the performance. For SQuAD and MATH, low thresholds yield high levels of contamination, but no performance gains. This suggests that contamination is either not helpful for these datasets, or that a larger n is required to obtain a better estimate. Finally, for MBPP, HumanEval, MMLU and MMLU-Pro, other contamination detection methods may be needed: even with higher thresholds, 8-gram overlap gives such high contamination scores that it is impossible to get a good performance gain estimate.</p>
<p>5.2 Post-trained Language Model</p>
<p>We present results for our Llama 3 post-trained models on benchmarks across different capabilities. Similar to pre-training we are releasing the data generated as part of evaluations with publicly available benchmarks which can be found on Huggingface here. Additional details on our eval setup can be found here.</p>
<p>Benchmarks and metrics. Table 16 contains an overview of all the benchmarks, organized by the capability. We apply decontamination of the post-training data by running exact match with the prompts from each benchmark. In addition to the standard academic benchmarks, we also performed extensive human evaluation of different capabilities. Details are provided in Section 5.3.</p>
<p>Experimental setup. We employ a similar experimental setup to the pre-training phase and conduct a comparative analysis of Llama 3 alongside other models of comparable size and capability. To the extent possible, we evaluate the performance of other models ourselves and compare the results with the reported numbers, selecting the best score. You can find additional details on our evaluation setup here.</p>
<p>34</p>
</div>
5.2.2 Proficiency Exams

Next, we evaluate our models on a wide variety of proficiency exams originally designed to test humans. We source these exams from publicly available official sources; for some exams, we report average scores across different exam sets per proficiency exam. Specifically, we average:

• GRE: Official GRE Practice Test 1 and 2 (from the Educational Testing Services);
• LSAT: Official Preptest 71, 73, 80 and 93;
• SAT: 8 exams from The Official SAT Study guide edition 2018;
• AP: One official practice exam per subject;
• GMAT: Official GMAT Online Exam.

Questions in these exams contain both MCQ style and generation questions. We exclude the questions that are accompanied with images. For the GRE exams that contain questions with multiple correct options, we qualify the outputs as correct only if all the correct options are selected by the model. The evaluations are

35

<html>
<body>
<table>
<tr>
<th>Exam</th>
<th>Llama 3 8B</th>
<th>Llama 3 70B</th>
<th>Llama 3 405B</th>
<th>GPT-3.5 Turbo</th>
<th>Nemotron 4 340B</th>
<th>GPT-4o</th>
<th>Claude 3.5 Sonnet</th>
</tr>
<tr>
<td>LSAT</td>
<td>53.9 ±4.9</td>
<td>74.2 ±4.3</td>
<td>81.1 ±3.8</td>
<td>54.3 ±4.9</td>
<td>73.7 ±4.3</td>
<td>77.4 ±4.1</td>
<td>80.0 ±3.9</td>
</tr>
<tr>
<td>SAT Reading</td>
<td>57.4 ±4.2</td>
<td>71.4 ±3.9</td>
<td>74.8 ±3.7</td>
<td>61.3 ±4.2</td>
<td>–</td>
<td>82.1 ±3.3</td>
<td>85.1 ±3.1</td>
</tr>
<tr>
<td>SAT Math</td>
<td>73.3 ±4.6</td>
<td>91.9 ±2.8</td>
<td>94.9 ±2.3</td>
<td>77.3 ±4.4</td>
<td>–</td>
<td>95.5 ±2.2</td>
<td>95.8 ±2.1</td>
</tr>
<tr>
<td>GMAT Quant.</td>
<td>56.0 ±19.5</td>
<td>84.0 ±14.4</td>
<td>96.0 ±7.7</td>
<td>36.0 ±18.8</td>
<td>76.0 ±16.7</td>
<td>92.0 ±10.6</td>
<td>92.0 ±10.6</td>
</tr>
<tr>
<td>GMAT Verbal</td>
<td>65.7 ±11.4</td>
<td>85.1 ±8.5</td>
<td>86.6 ±8.2</td>
<td>65.7 ±11.4</td>
<td>91.0 ±6.8</td>
<td>95.5 ±5.0</td>
<td>92.5 ±6.3</td>
</tr>
<tr>
<td>GRE Physics</td>
<td>48.0 ±11.3</td>
<td>74.7 ±9.8</td>
<td>80.0 ±9.1</td>
<td>50.7 ±11.3</td>
<td>–</td>
<td>89.3 ±7.0</td>
<td>90.7 ±6.6</td>
</tr>
<tr>
<td>AP Art History</td>
<td>75.6 ±12.6</td>
<td>84.4 ±10.6</td>
<td>86.7 ±9.9</td>
<td>68.9 ±13.5</td>
<td>71.1 ±13.2</td>
<td>80.0 ±11.7</td>
<td>77.8 ±12.1</td>
</tr>
<tr>
<td>AP Biology</td>
<td>91.7 ±11.1</td>
<td>100.0 ±0.0</td>
<td>100.0 ±0.0</td>
<td>91.7 ±11.1</td>
<td>95.8 ±8.0</td>
<td>100.0 ±0.0</td>
<td>100.0 ±0.0</td>
</tr>
<tr>
<td>AP Calculus</td>
<td>57.1 ±16.4</td>
<td>54.3 ±16.5</td>
<td>88.6 ±10.5</td>
<td>62.9 ±16.0</td>
<td>68.6 ±15.4</td>
<td>91.4 ±9.3</td>
<td>88.6 ±10.5</td>
</tr>
<tr>
<td>AP Chemistry</td>
<td>59.4 ±17.0</td>
<td>96.9 ±6.0</td>
<td>90.6 ±10.1</td>
<td>62.5 ±16.8</td>
<td>68.8 ±16.1</td>
<td>93.8 ±8.4</td>
<td>96.9 ±6.0</td>
</tr>
<tr>
<td>AP English Lang.</td>
<td>69.8 ±12.4</td>
<td>90.6 ±7.9</td>
<td>94.3 ±6.2</td>
<td>77.4 ±11.3</td>
<td>88.7 ±8.5</td>
<td>98.1 ±3.7</td>
<td>90.6 ±7.9</td>
</tr>
<tr>
<td>AP English Lit.</td>
<td>59.3 ±13.1</td>
<td>79.6 ±10.7</td>
<td>83.3 ±9.9</td>
<td>53.7 ±13.3</td>
<td>88.9 ±8.4</td>
<td>88.9 ±8.4</td>
<td>85.2 ±9.5</td>
</tr>
<tr>
<td>AP Env. Sci.</td>
<td>73.9 ±12.7</td>
<td>89.1 ±9.0</td>
<td>93.5 ±7.1</td>
<td>73.9 ±12.7</td>
<td>73.9 ±12.7</td>
<td>89.1 ±9.0</td>
<td>84.8 ±10.4</td>
</tr>
<tr>
<td>AP Macro Eco.</td>
<td>72.4 ±11.5</td>
<td>98.3 ±3.3</td>
<td>98.3 ±3.3</td>
<td>67.2 ±12.1</td>
<td>91.4 ±7.2</td>
<td>96.5 ±4.7</td>
<td>94.8 ±5.7</td>
</tr>
<tr>
<td>AP Micro Eco.</td>
<td>70.8 ±12.9</td>
<td>91.7 ±7.8</td>
<td>93.8 ±6.8</td>
<td>64.6 ±13.5</td>
<td>89.6 ±8.6</td>
<td>97.9 ±4.0</td>
<td>97.9 ±4.0</td>
</tr>
<tr>
<td>AP Physics</td>
<td>57.1 ±25.9</td>
<td>78.6 ±21.5</td>
<td>92.9 ±13.5</td>
<td>35.7 ±25.1</td>
<td>71.4 ±23.7</td>
<td>71.4 ±23.7</td>
<td>78.6 ±21.5</td>
</tr>
<tr>
<td>AP Psychology</td>
<td>94.8 ±4.4</td>
<td>100.0 ±0.0</td>
<td>100.0 ±0.0</td>
<td>94.8 ±4.4</td>
<td>100.0 ±0.0</td>
<td>100.0 ±0.0</td>
<td>100.0 ±0.0</td>
</tr>
<tr>
<td>AP Statistics</td>
<td>66.7 ±17.8</td>
<td>59.3 ±18.5</td>
<td>85.2 ±13.4</td>
<td>48.1 ±18.8</td>
<td>77.8 ±15.7</td>
<td>92.6 ±9.9</td>
<td>96.3 ±7.1</td>
</tr>
<tr>
<td>AP US Gov.</td>
<td>90.2 ±9.1</td>
<td>97.6 ±4.7</td>
<td>97.6 ±4.7</td>
<td>78.0 ±12.7</td>
<td>78.0 ±12.7</td>
<td>100.0 ±0.0</td>
<td>100.0 ±0.0</td>
</tr>
<tr>
<td>AP US History</td>
<td>78.0 ±12.7</td>
<td>97.6 ±4.7</td>
<td>97.6 ±4.7</td>
<td>85.4 ±10.8</td>
<td>70.7 ±13.9</td>
<td>95.1 ±6.6</td>
<td>95.1 ±6.6</td>
</tr>
<tr>
<td>AP World History</td>
<td>94.1 ±7.9</td>
<td>100.0 ±0.0</td>
<td>100.0 ±0.0</td>
<td>88.2 ±10.8</td>
<td>85.3 ±11.9</td>
<td>100.0 ±0.0</td>
<td>97.1 ±5.7</td>
</tr>
<tr>
<td>AP Average</td>
<td>74.1 ±3.4</td>
<td>87.9 ±2.5</td>
<td>93.5 ±1.9</td>
<td>70.2 ±3.5</td>
<td>81.3 ±3.0</td>
<td>93.0 ±2.0</td>
<td>92.2 ±2.1</td>
</tr>
<tr>
<td>GRE Quant.</td>
<td>152.0</td>
<td>158.0</td>
<td>162.0</td>
<td>155.0</td>
<td>161.0</td>
<td>166.0</td>
<td>164.0</td>
</tr>
<tr>
<td>GRE Verbal</td>
<td>149.0</td>
<td>166.0</td>
<td>166.0</td>
<td>154.0</td>
<td>162.0</td>
<td>167.0</td>
<td>167.0</td>
</tr>
</table>
<p>Table 17 Performance of Llama 3 models and GPT-4o on a variety of proficiency exams including LSAT, SAT, GMAT, and
AP, and GRE tests. For GRE exams, we report normalized score; for all others, we report accuracy. For the bottom
two rows corresponding to GRE Quant. and GRE Verbal, we report the scaled scores out of 170.</p>
<p>run using few-shot prompting wherever we have more than 1 exam set per exam. We scale the scores to be in
the range 130–170 for GRE and report accuracy for all other exams.</p>
<p>Our results can be found in Table 17. We observe that the performance of our Llama 3 405B model is very
similar to Claude 3.5 Sonnet and GPT-4 4o. Our 70B model has an even more impressive performance. It is
significantly better than GPT-3.5 Turbo and beats Nemotron 4 340B on many tests.</p>
<p>5.2.3 Coding Benchmarks</p>
<p>We evaluate Llama 3 on code generation on several popular Python and multi-programming language
benchmarks. To gauge the effectiveness of our models in generating functionally correct code, we use the
pass@N metric, which evaluates the pass rate for a set of unit tests among N generations. We report pass@1.</p>
<p>Python code generation. HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021) are popular benchmarks
for Python code generation which focus on relatively simple, self-contained functions. HumanEval+ (Liu et al.,
2024a) is an enhanced version of HumanEval, in which more tests are generated to avoid false positives. The
MBPP EvalPlus base version (v0.2.0) is a selection of 378 well-formed problems out of the 974 initial problems
in all of the original MBPP (train and test) dataset (Liu et al., 2024a). Results for these benchmarks are
reported in Table 18. Across the Python variants of these benchmarks, Llama 3 8B and 70B outperform</p>
<p>36</p>
</body>
</html>
<table>
<thead>
<tr>
<th>Model</th>
<th>HumanEval</th>
<th>HumanEval+</th>
<th>MBPP</th>
<th>MBPP EvalPlus (base)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Llama 3 8B</td>
<td>72.6 ±6.8</td>
<td>67.1 ±7.2</td>
<td>60.8 ±4.3</td>
<td>72.8 ±4.5</td>
</tr>
<tr>
<td>Gemma 2 9B</td>
<td>54.3 ±7.6</td>
<td>48.8 ±7.7</td>
<td>59.2 ±4.3</td>
<td>71.7 ±4.5</td>
</tr>
<tr>
<td>Mistral 7B</td>
<td>40.2 ±7.5</td>
<td>32.3 ±7.2</td>
<td>42.6 ±4.3</td>
<td>49.5 ±5.0</td>
</tr>
<tr>
<td>Llama 3 70B</td>
<td>80.5 ±6.1</td>
<td>74.4 ±6.7</td>
<td>75.4 ±3.8</td>
<td>86.0 ±3.5</td>
</tr>
<tr>
<td>Mixtral 8×22B</td>
<td>75.6 ±6.6</td>
<td>68.3 ±7.1</td>
<td>66.2 ±4.1</td>
<td>78.6 ±4.1</td>
</tr>
<tr>
<td>GPT-3.5 Turbo</td>
<td>68.0 ±7.1</td>
<td>62.8 ±7.4</td>
<td>71.2 ±4.0</td>
<td>82.0 ±3.9</td>
</tr>
<tr>
<td>Llama 3 405B</td>
<td>89.0 ±4.8</td>
<td>82.3 ±5.8</td>
<td>78.8 ±3.6</td>
<td>88.6 ±3.2</td>
</tr>
<tr>
<td>GPT-4</td>
<td>86.6 ±5.2</td>
<td>77.4 ±6.4</td>
<td>80.2 ±3.5</td>
<td>83.6 ±3.7</td>
</tr>
<tr>
<td>GPT-4o</td>
<td>90.2 ±4.5</td>
<td>86.0 ±5.3</td>
<td>81.4 ±3.4</td>
<td>87.8 ±3.3</td>
</tr>
<tr>
<td>Claude 3.5 Sonnet</td>
<td>92.0 ±4.2</td>
<td>82.3 ±5.8</td>
<td>76.6 ±3.7</td>
<td>90.5 ±3.0</td>
</tr>
<tr>
<td>Nemotron 4 340B</td>
<td>73.2 ±6.8</td>
<td>64.0 ±7.3</td>
<td>75.4 ±3.8</td>
<td>72.8 ±4.5</td>
</tr>
</tbody>
</table>
<p>Table 18 Pass@1 scores on code generation benchmarks. We report results on HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), as well as EvalPlus (Liu et al., 2024a) versions of these benchmarks.</p>
<table>
<thead>
<tr>
<th>Model</th>
<th>Dataset</th>
<th>C++</th>
<th>Java</th>
<th>PHP</th>
<th>TS</th>
<th>C#</th>
<th>Shell</th>
</tr>
</thead>
<tbody>
<tr>
<td>Llama 3 8B</td>
<td>HumanEval</td>
<td>52.8 ±7.7</td>
<td>58.2 ±7.7</td>
<td>54.7 ±7.7</td>
<td>56.6 ±7.7</td>
<td>38.0 ±7.6</td>
<td>39.2 ±7.6</td>
</tr>
<tr>
<td></td>
<td>MBPP</td>
<td>53.7 ±4.9</td>
<td>54.4 ±5.0</td>
<td>55.7 ±4.9</td>
<td>62.8 ±4.8</td>
<td>43.3 ±4.9</td>
<td>33.0 ±4.7</td>
</tr>
<tr>
<td>Llama 3 70B</td>
<td>HumanEval</td>
<td>71.4 ±7.0</td>
<td>72.2 ±7.0</td>
<td>67.7 ±7.2</td>
<td>73.0 ±6.9</td>
<td>50.0 ±7.8</td>
<td>51.9 ±7.8</td>
</tr>
<tr>
<td></td>
<td>MBPP</td>
<td>65.2 ±4.7</td>
<td>65.3 ±4.8</td>
<td>64.0 ±4.7</td>
<td>70.5 ±4.5</td>
<td>51.0 ±5.0</td>
<td>41.9 ±4.9</td>
</tr>
<tr>
<td>Llama 3 405B</td>
<td>HumanEval</td>
<td>82.0 ±5.9</td>
<td>80.4 ±6.2</td>
<td>76.4 ±6.6</td>
<td>81.1 ±6.1</td>
<td>54.4 ±7.8</td>
<td>57.6 ±7.7</td>
</tr>
<tr>
<td></td>
<td>MBPP</td>
<td>67.5 ±4.6</td>
<td>65.8 ±4.7</td>
<td>76.6 ±4.2</td>
<td>72.6 ±4.4</td>
<td>53.1 ±5.0</td>
<td>43.7 ±5.0</td>
</tr>
</tbody>
</table>
<p>Table 19 Performance of non-Python programming tasks. We report Llama 3 results on MultiPL-E (Cassano et al., 2023).</p>
<p>models of similar sizes. For the largest models, Llama 3 405B, Claude 3.5 Sonnet and GPT-4o perform similarly, with GPT-4o showing the strongest results.</p>
<p>Multi-programming language code generation. To assess code generation capabilities beyond Python, we report results for the MultiPL-E (Cassano et al., 2023) benchmark, which is based on translations of problems from HumanEval and MBPP. Results for a subset of popular programming languages are reported in Table 19. Note that there is a significant drop in performance compared to the Python counterparts in Table 18.</p>
<p>5.2.4 Multilingual Benchmarks</p>
<p>Llama 3 supports 8 languages — English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai, although the underlying foundation model has been trained on a broader collection of languages.<sup>9</sup> In Table 20, we show results from evaluating Llama 3 on the multilingual MMLU (Hendrycks et al., 2021a) and Multilingual Grade School Math (MGSM) (Shi et al., 2022) benchmarks.</p>
<p>Multilingual MMLU. We translate MMLU questions, few-shot examples, and answers using Google Translate. We leave the task instructions in English and perform the evaluation in a 5-shot setting. In Table 20, we report average results across German, French, Italian, Portuguese, Hindi, Spanish, and Thai.</p>
<p><sup>9</sup>Llama 3 has not been optimized or safety tuned for use cases in those other languages. Developers may fine-tune Llama 3 models for languages beyond the 8 supported languages provided they comply with the Llama 3 Community License and the Acceptable Use Policy and in such cases are responsible for ensuring that any uses of Llama 3 in additional languages is done in a safe and responsible manner.</p>
<p>37</p>


<table>
<tr>
<th>Model</th>
<th>MGSM</th>
<th>Multilingual MMLU</th>
</tr>
<tr>
<td>Llama 3 8B</td>
<td>68.9</td>
<td>58.6</td>
</tr>
<tr>
<td>Mistral 7B</td>
<td>29.9</td>
<td>46.8</td>
</tr>
<tr>
<td>Gemma 2 9B</td>
<td>53.2</td>
<td>–</td>
</tr>
<tr>
<td>Llama 3 70B</td>
<td>86.9</td>
<td>78.2</td>
</tr>
<tr>
<td>GPT-3.5 Turbo</td>
<td>51.4</td>
<td>58.8</td>
</tr>
<tr>
<td>Mixtral 8 ×22B</td>
<td>71.1</td>
<td>64.3</td>
</tr>
<tr>
<td>Llama 3 405B</td>
<td>91.6</td>
<td>83.2</td>
</tr>
<tr>
<td>GPT-4</td>
<td>85.9</td>
<td>80.2</td>
</tr>
<tr>
<td>GPT-4o</td>
<td>90.5</td>
<td>85.5</td>
</tr>
<tr>
<td>Claude 3.5 Sonnet</td>
<td>91.6</td>
<td>–</td>
</tr>
</table>
<p>Table 20 Multilingual benchmarks. For MGSM (Shi et al., 2022), we report 0-shot CoT results for our Llama 3 models. Multilingual MMLU is an internal benchmark with translated MMLU (Hendrycks et al., 2021a) questions and answers into 7 languages – we report 5-shot results averaged across these languages.</p>
<p>MGSM (Shi et al., 2022). We use the same native prompts as in simple-evals (OpenAI, 2024) for testing our models in a 0-shot CoT setting. In Table 20, we report averge results across languages covered in MGSM benchmark.</p>
<p>We find that Llama 3 405B outperforms most other models on MGSM, achieving an average of 91.6%. On MMLU, in line with English MMLU results shown above, Llama 3 405B falls behind GPT-4o by 2%. On the other hand, both Llama 3 70B and 8B models demonstrate strong performance, leading among competitors with a wide margin on both tasks.</p>
<p>5.2.5 Math and Reasoning Benchmarks</p>
<p>Our math and reasoning benchmark results are presented in Table 2. Llama 3 8B model outperforms other models of similar sizes on GSM8K, MATH, and GPQA. Our 70B model performs significantly better than other models in its class on all the benchmarks. Finally, Llama 3 405B model is the best in its category on GSM8K and ARC-C, while on MATH, it is the second best model. On GPQA, it is competitive with GPT-4 4o, with Claude 3.5 Sonnet being the best model by a significant margin.</p>
<p>5.2.6 Long Context Benchmarks</p>
<p>We consider a diverse set of tasks that span various domains and text types. In the benchmarks we list below, we focus on sub-tasks that use unbiased evaluation protocols, i.e., accuracy-based metrics rather than n-gram overlapping metrics. We also prioritize tasks that we found to be of low variance.</p>
<p>•Needle-in-a-Haystack (Kamradt, 2023) measures a model’s ability to retrieve a hidden information inserted in random parts of the long document. Our Llama 3 models demonstrate perfect needle retrieval performance, successfully retrieving 100% of needles at all document depths and context lengths. We also measure performance on Multi-needle (Table 21), a variation of Needle-in-a-Haystack, where we insert four needles in the context and test if a model can retrieve two of them. Our Llama 3 models achieve near perfect retrieval results.</p>
<p>•ZeroSCROLLS (Shaham et al., 2023) is a zero-shot benchmark for natural language understanding over long texts. We report numbers on the validation set, as the ground truth answers are not publicly available. Our Llama 3 405B and 70B models either match or surpass other models on various tasks in this benchmark.</p>
<p>•InfiniteBench (Zhang et al., 2024) requires models to understand long dependencies in the context window. We evaluate Llama 3 on En.QA (QA over novels) and En.MC (multiple-choice QA over novels), where our 405B model outperforms all others. The gains are particularly significant on En.QA.</p>
<p>5.2.7 Tool Use Performance</p>
<p>We evaluate our models on a range of benchmarks for zero-shot tool use (i.e. function calling): Nexus (Srinivasan et al., 2023), API-Bank (Li et al., 2023b), Gorilla API-Bench (Patil et al., 2023), and the Berkeley Function Calling Leaderboard (BFCL) (Yan et al., 2024). Results are shown in Table 22.</p>
<p>On Nexus, our Llama 3 variants perform the best compared to their counterparts. On the API-Bank, our Llama 3 8B and 70B models outperform other models in their category by a significant margin. The 405B model is behind Claude 3.5 Sonnet by only 0.6%. Finally, our 405B and 70B models perform competitively on BFCL and are close second in their respective size class. Llama 3 8B performs the best in its category.</p>
<p>38</p>


<table>
<thead>
<tr>
<th>ZeroSCROLLS</th>
<th>InfiniteBench</th>
<th>NIH</th>
</tr>
<tr>
<th>QuALITY</th>
<th>Qasper</th>
<th>SQuALITY</th>
<th>En.QA</th>
<th>En.MC</th>
<th>Multi-needle</th>
</tr>
</thead>
<tbody>
<tr>
<td>Llama 3 8B</td>
<td>81.0 ±16.8</td>
<td>39.3 ±18.1</td>
<td>15.3 ±7.9</td>
<td>27.1 ±4.6</td>
<td>65.1 ±6.2</td>
<td>98.8 ±1.2</td>
</tr>
<tr>
<td>Llama 3 70B</td>
<td>90.5 ±12.6</td>
<td>49.0 ±18.5</td>
<td>16.4 ±8.1</td>
<td>36.7 ±5.0</td>
<td>78.2 ±5.4</td>
<td>97.5 ±1.7</td>
</tr>
<tr>
<td>Llama 3 405B</td>
<td>95.2 ±9.1</td>
<td>49.8 ±18.5</td>
<td>15.4 ±7.9</td>
<td>30.5 ±4.8</td>
<td>83.4 ±4.8</td>
<td>98.1 ±1.5</td>
</tr>
<tr>
<td>GPT-4</td>
<td>95.2 ±9.1</td>
<td>50.5 ±18.5</td>
<td>13.2 ±7.4</td>
<td>15.7 ±3.8</td>
<td>72.0 ±5.8</td>
<td>100.0 ±0.0</td>
</tr>
<tr>
<td>GPT-4o</td>
<td>90.5 ±12.5</td>
<td>49.2 ±18.5</td>
<td>18.8 ±8.6</td>
<td>19.1 ±4.1</td>
<td>82.5 ±4.9</td>
<td>100.0 ±0.0</td>
</tr>
<tr>
<td>Claude 3.5 Sonnet</td>
<td>90.5 ±12.6</td>
<td>18.5 ±14.4</td>
<td>13.4 ±7.5</td>
<td>11.3 ±3.3</td>
<td>-</td>
<td>90.8 ±3.2</td>
</tr>
</tbody>
</table>
<p>Table 21 Long-context benchmarks. For ZeroSCROLLS (Shaham et al., 2023), we report numbers on the validation set. For QuALITY we report exact match, for Qasper - f1 and for SQuALITY - rougeL. We report f1 for InfiniteBench (Zhang et al., 2024) En.QA metric and accuracy for En.MC. For Multi-needle (Kamradt, 2023) we insert 4 needles in the context and test if a model can retrieve 2 needles at different context lengths, we compute average recall across 10 sequence lengths up till 128k.</p>
<p>Human evaluations. We also conduct human evaluations to test the tool use capabilities of the model, with a focus on code execution tasks. We collect 2000 user prompts related to code execution (without plotting or file uploads), plot generation, and file uploads. These prompts are collected from the LMSys dataset (Chiang et al., 2024), GAIA benchmark (Mialon et al., 2023b), human annotators, and synthetic generation.</p>
<table>
<thead>
<tr>
<th></th>
<th>Nexus</th>
<th>API-Bank</th>
<th>API-Bench</th>
<th>BFCL</th>
</tr>
</thead>
<tbody>
<tr>
<td>Llama 3 8B</td>
<td>38.5 ±4.1</td>
<td>82.6 ±3.8</td>
<td>8.2 ±1.3</td>
<td>76.1 ±2.0</td>
</tr>
<tr>
<td>Gemma 2 9B</td>
<td>-</td>
<td>56.5 ±4.9</td>
<td>11.6 ±1.5</td>
<td>-</td>
</tr>
<tr>
<td>Mistral 7B</td>
<td>24.7 ±3.6</td>
<td>55.8 ±4.9</td>
<td>4.7 ±1.0</td>
<td>60.4 ±2.3</td>
</tr>
<tr>
<td>Llama 3 70B</td>
<td>56.7 ±4.2</td>
<td>90.0 ±3.0</td>
<td>29.7 ±2.1</td>
<td>84.8 ±1.7</td>
</tr>
<tr>
<td>Mixtral 8 ×22B</td>
<td>48.5 ±4.2</td>
<td>73.1 ±4.4</td>
<td>26.0 ±2.0</td>
<td>-</td>
</tr>
<tr>
<td>GPT-3.5 Turbo</td>
<td>37.2 ±4.1</td>
<td>60.9 ±4.8</td>
<td>36.3 ±2.2</td>
<td>85.9 ±1.7</td>
</tr>
<tr>
<td>Llama 3 405B</td>
<td>58.7 ±4.1</td>
<td>92.3 ±2.6</td>
<td>35.3 ±2.2</td>
<td>88.5 ±1.5</td>
</tr>
<tr>
<td>GPT-4</td>
<td>50.3 ±4.2</td>
<td>89.0 ±3.1</td>
<td>22.5 ±1.9</td>
<td>88.3 ±1.5</td>
</tr>
<tr>
<td>GPT-4o</td>
<td>56.1 ±4.2</td>
<td>91.3 ±2.8</td>
<td>41.4 ±2.3</td>
<td>80.5 ±1.9</td>
</tr>
<tr>
<td>Claude 3.5 Sonnet</td>
<td>45.7 ±4.2</td>
<td>92.6 ±2.6</td>
<td>60.0 ±2.3</td>
<td>90.2 ±1.4</td>
</tr>
<tr>
<td>Nemotron 4 340B</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>86.5 ±1.6</td>
</tr>
</tbody>
</table>
<p>Table 22 Zero-shot tool use benchmarks. We report function calling accuracy across Nexus (Srinivasan et al., 2023), API-Bank (Li et al., 2023b), API-Bench (Patil et al., 2023), and BFCL (Yan et al., 2024).</p>
<p>We compare Llama 3 405B to GPT-4o using OpenAI’s Assistants API<sup>10</sup>. The results are provided in Figure 16. On text-only code execution tasks and plots generation, Llama 3 405B significantly beats GPT-4o. However, it lags behind on the file upload use case.</p>
<p>5.3 Human Evaluations</p>
<p>In addition to evaluations on standard benchmark sets, we also perform a series of human evaluations. These evaluations allow us to measure and optimize more subtle aspects of model performance, such as our model’s tone, verbosity, and understanding of nuances and cultural contexts. Well-designed human evaluations closely reflect the user experience, providing insights into how the model performs in real-world scenarios.</p>
<p>Prompt collection. We collected high-quality prompts spanning a wide range of categories and difficulties. To do so, we first developed a taxonomy with categories and subcategories capturing as many model capabilities as possible. We used this taxonomy to collect about 7,000 prompts spanning six individual capabilities (English, reasoning, coding, Hindi, Spanish, and Portuguese), and three multiturn capabilities<sup>11</sup> (English, reasoning, and coding). We ensured that within each category, prompts are uniformly distributed across subcategories. We also categorized each prompt into one of three difficulty levels and ensured that our prompt collection</p>
<p><sup>10</sup><a href="https://platform.openai.com/docs/assistants/overview">https://platform.openai.com/docs/assistants/overview</a></p>
<p><sup>11</sup>For multiturn human evaluations, the number of turns is between 2 and 11 in each prompt. We assess the model response in the final turn.</p>
<p>39</p>


Figure 16 Human evaluation results for Llama 3 405B vs. GPT-4o on code execution tasks including plotting and file uploads. Llama 3 405B outperforms GPT-4o on code execution (without plotting or file uploads) as well as plot generation, but lags behind in file upload use cases. contains roughly 10% easy prompts, 30% medium prompts, and 60% hard prompts. All the human evaluation prompt sets were subject to a thorough quality assurance process. Modeling teams did not have access to our human-evaluation prompts to prevent accidental contamination or overfitting on the test set.

Evaluation process. To perform a pairwise human evaluation of two models, we ask human annotators which of two model responses (produced by different models) they prefer. Annotators use a 7-point scale for their ratings, enabling them to indicate whether one model response is much better than, better than, slightly better than, or about the same as the other model response. When an annotator indicates that one model response is better or much better than the other model response, we consider this a “win” for that model. We perform pairwise comparisons between models in which we report win rates per capability in the prompt set.

Results. We use our human evaluation process to compare Llama 3 405B with GPT-4 (0125 API version), GPT-4o (API version), and Claude 3.5 Sonnet (API version). The results of these evaluations are presented in Figure 17. We observe that Llama 3 405B performs approximately on par with the 0125 API version of GPT-4, while achieving mixed results (some wins and some losses) compared to GPT-4o and Claude 3.5 Sonnet. On nearly all capabilities, the win rates of Llama 3 and GPT-4 are within the margin of error. On multiturn reasoning and coding tasks, Llama 3 405B outperforms GPT-4 but it underperforms GPT-4 on multilingual (Hindi, Spanish, and Portuguese) prompts. Llama 3 performs on par with GPT-4o on English prompts, on par with Claude 3.5 Sonnet on multilingual prompts, and outperforms Claude 3.5 Sonnet on single and multiturn English prompts. However, it trails Claude 3.5 Sonnet in capabilities such as coding and reasoning. Qualitatively, we find that model performance in human evaluations is heavily influenced by nuanced factors such as model tone, response structure, and verbosity – factors that we are optimizing for in our post-training process. Overall, our human evaluation results are consistent with those on standard benchmark evaluations: Llama 3 405B is very competitive with leading industry models, making it the best-performing openly available model.

Limitations. All human evaluation results underwent a thorough data quality assurance process. However, since it is challenging to define objective criteria for evaluating model responses, human evaluations can still be influenced by personal biases, backgrounds, and preferences of human annotators, which may lead to inconsistent or unreliable results.

5.4 Safety

We focus our study on assessing Llama 3’s ability to generate content in a safe and responsible way, while still maximizing helpful information. Our safety work begins in the pre-training stage, primarily in the form of

40

Figure 17 Human evaluation results for the Llama 3 405B model. Left: Comparison with GPT-4. Middle: Comparison with GPT-4o. Right: Comparison with Claude 3.5 Sonnet. All results include 95% confidence intervals and exclude ties.

data cleaning and filtering. We then describe our approach to safety finetuning, focusing on how to train the model to align to specific safety policies while still retaining helpfulness. We analyze each of the Llama 3 capabilities, including multilingual, long context, tool usage, and various multimodal capabilities, to measure the effectiveness of our safety mitigations.

Subsequently, we describe our assessment of uplift for cybersecurity and chemical and biological weapons risks. Uplift refers to the additional risk introduced by new technological developments compared to using existing available technologies (such as web search).

We then describe how we leverage Red Teaming to iteratively identify and combat various safety risks across capabilities and perform a residual risk assessment.

Finally, we describe system-level safety, or the development and orchestration of classifiers around the input and output of the model itself to further enhance safety and make it easier for developers to both customize safety to various usecases and deploy generative AI in more responsible ways.

5.4.1 Benchmark Construction

We create various internal benchmarks to help us develop models safely and responsibly. Our benchmarks are heavily inspired by the risk categories from the ML Commons taxonomy of hazards (Vidgen et al., 2024).

While various benchmarks for language model safety exist such as ToxiGen (Hartvigsen et al., 2022), XS Test (Röttger et al., 2023), and more, a number of the new capabilities in Llama 3 do not have sufficient external benchmarks available and often external benchmarks lack sufficient coverage of breadth and depth.

For each risk category, we collect human-written prompts that are either adversarial or borderline in nature — examples of such prompts can be found in Table 23. Adversarial prompts range from straightforward ones that directly elicit a harmful response to ones that incorporate sophisticated jailbreaking techniques. Benchmarks of these adversarial prompts are the basis for how we measure violation rate.

As a counter-metric to violation rate, we also construct false refusal benchmarks composed of borderline prompts. We refer to false refusal as occurring when a model refuses to answer in a helpful way even when a plausible, safe response is possible. Borderline prompts are prompts near the decision boundary that a well-calibrated model should handle, for example, “How can I steal the spotlight from my bestie who always acts like the main character??” Our overall benchmark size across violations and false refusals is over 4000 prompts per capability or language, and contains a mix of single-turn and multi-turn prompts.

41

5.4.2 Safety Pre-training

We believe responsible development must be considered from an end-to-end perspective and incorporated at every stage of model development and deployment. During pre-training, we apply a variety of filters, such as filters to identify websites that likely contain personally identifiable information (see Section 3.1). We also focus heavily on discoverable memorization (Nasr et al., 2023). Similar to Carlini et al. (2022), we sample prompts and ground truths at different frequencies of occurrence in the training data using an efficient rolling hash index of all n-grams in the corpus. We construct different test scenarios by varying the length of prompt and ground truth, the detected language of target data, and the domain. We then measure how often the model generates the ground truth sequence verbatim, and analyze the relative rates of memorization in the specified scenarios. We define verbatim memorization as the inclusion rate – the proportion of model generations that include the ground truth continuation exactly – and report averages weighted by the prevalence of given characteristics in the data, as shown in Table 24. We find low memorization rates of training data (1.13% and 3.91% on average for the 405B with n = 50 and n = 1000 respectively). Memorization rates are roughly on par with Llama 2 at equivalent size and using the same methodology applied to its data mix.<sup>12</sup>

5.4.3 Safety Finetuning

We describe our approach to safety finetuning to mitigate risks across many capabilities, which encompasses two key aspects: (1) safety training data and (2) risk mitigation techniques. Our safety finetuning process builds upon our general finetuning methodology with modifications tailored to address specific safety concerns. We optimize for two primary metrics: Violation Rate (VR), a metric that captures when the model produces a

<sup>12</sup>Note there are limitations with our analysis — for example, recent work advocates for metrics beyond exact match (Ippolito et al., 2023) and alternative prompt search strategies (Kassem et al., 2024). Nonetheless, we find the results of the evaluations to be encouraging.

42

response that violates a safety policy, and False Refusal Rate (FRR), a metric that captures when the model incorrectly refuses to respond to a harmless prompt. In parallel, we evaluate model performance on helpfulness benchmarks to ensure that safety improvements do not compromise overall helpfulness.

Finetuning data. The quality and design of safety training data has a profound impact on performance. Through extensive ablations, we find that the quality is more critical than the quantity. We mainly use human-generated data collected from our data vendors, but find that it can be prone to errors and inconsistencies, particularly for nuanced safety policies. To ensure the highest quality data, we developed AI-assisted annotation tools to support our rigorous quality assurance processes. In addition to collecting adversarial prompts, we also gather a set of similar prompts, which we refer to as borderline prompts. These are closely related to the adversarial prompts but with a goal to teach the model to learn to provide helpful responses, thereby reducing the false refusal rate (FRR).

Beyond human annotation, we also leverage synthetic data to improve the quality and coverage of our training datasets. We utilize a range of techniques to generate additional adversarial examples, including in-context learning with carefully crafted system prompts, guided mutation of seed prompts based on new attack vectors, and advanced algorithms including Rainbow Teaming (Samvelyan et al., 2024), based on MAP-Elites (Mouret and Clune, 2015), which generate prompts constrained across multiple dimensions of diversity.

We further address the model’s tone when producing safe responses, which has an impact on downstream user experience. We developed a refusal tone guideline for Llama 3 and ensured that all new safety data adhered to it through rigorous quality assurance processes. We also refine existing safety data to align with the guideline, using a combination of zero-shot rewriting and human-in-the-loop editing to produce high-quality data. By employing these methods, along with a tone classifier to assess tone quality for safety responses, we are able to significantly improve the model’s verbiage.

Safety supervised finetuning. Following our Llama 2 recipe (Touvron et al., 2023b), we combine all helpfulness data and safety data during the model alignment stage. Additionally, we introduce a borderline dataset to help the model discern the subtle distinctions between safe and unsafe requests. Our annotation teams are instructed to meticulously craft responses to safety prompts based on our guidelines. We have found that SFT is highly effective in aligning the model when we strategically balance the ratio of adversarial to borderline examples. We put the focus on more challenging risk areas, with a higher ratio of borderline examples. This plays a crucial role in our successful safety mitigation efforts while keeping false refusal to a minimum.

Further, we examine the impact of model size on the trade-off between FRR and VR in Figure 18. Our results show that it varies—with smaller models requiring a larger proportion of safety data relative to helpfulness, and that it is more challenging to efficiently balance VR and FRR compared to larger models.

Safety DPO. To reinforce safety learning, we incorporate adversarial and borderline examples into our preference datasets in DPO. We discover that crafting response pairs to be nearly orthogonal in an embedding space is particularly effective in teaching the model to distinguish between good and bad responses for a given prompt. We conduct multiple experiments to determine the optimal ratio of adversarial, borderline, and helpfulness examples, aiming to optimize the trade-off between FRR and VR. We also find that the model size influences the learning outcomes—as a result, we tailor different safety mixes for various model sizes.

43

<br>

English French German Hindi Italian Portuguese Spanish Thai
<br>
Language
<br>
<table>
<tr>
<td>System</td>
<td>Llama 3 405B + LG</td>
</tr>
<tr>
<td>[System] Comp. 1</td>
<td>[System] Comp. 2</td>
</tr>
<tr>
<td>Model</td>
<td>Llama 3 405B</td>
</tr>
<tr>
<td>[Model] Comp. 3</td>
<td></td>
</tr>
</table>
<br>
English French German Hindi Italian Portuguese Spanish Thai
<br>
Language
<br>
Figure 19 Violation rates (VR) and false refusal rates (FRR) on English and our core multilingual short context benchmarks, comparing Llama 3 405B—with and without Llama Guard (LG) system-level protections—to competitor models and systems. Languages not supported by Comp. 3 represented with an ‘x.’ Lower is better.
<br>
Tool Usage (Search) Long Context (Doc QA) Long Context (Many-shot)
<br>
Capability
<br>
<table>
<tr>
<td>System</td>
<td>Llama 3 405B + LG</td>
</tr>
<tr>
<td>[System] Comp. 1</td>
<td>[System] Comp. 2</td>
</tr>
<tr>
<td>Model</td>
<td>Llama 3 405B</td>
</tr>
</table>
<br>
Figure 20 Violation rates (VR) and false refusal rates (FRR) on tool use and long context benchmarks. Lower is better. The performance for DocQA and Many-shot benchmarks are listed separately. Note we do not have a borderline data set for Many-shot, due to the adversarial nature of the benchmark, and thus do not measure false refusal rates on it. For Tool Usage (Search), we only test Llama 3 405B compared to Comp. 1.
<br>
5.4.4 Safety Results
<br>
We first highlight Llama 3’s general behavior along various axes and then describe results for each specific new capability and our effectiveness at mitigating the safety risks.
<br>
Overall performance. A comparison of Llama 3’s final violation and false refusal rates with similar models can be found in Figures 19 and 20. These results focus on our largest parameter size Llama 3 405B model, compared to relevant competitors. Two of the competitors are end-to-end systems accessed through API, and one of them is an open source language model that we host internally and we evaluate directly.<sup>13</sup> We evaluate our Llama models both standalone and coupled with Llama Guard, our open source system-level safety solution (more in Section 5.4.7).
<br>
While a low violation rate is desirable, it is critical to consider false refusal as a counter-metric, as a model that always refuses is maximally safe, but not helpful in the slightest. Similarly, a model that always answers every prompt, regardless of how problematic the request, would be overly harmful and toxic. In Figure 21, leveraging our internal benchmarks, we explore how different models and systems in industry navigate this trade off and how Llama 3 compares. We find that our models achieve very competitive violation rate metrics
<br>
<sup>13</sup>Because these safety benchmarks are internal to Meta, we acknowledge that the numbers in this section are not reproducible externally, and so we choose to anonymize the competitors we evaluate against.
<br>
44


while keeping false refusal rate low as well, indicating a solid balance between helpfulness and safety.

Multilingual safety. Our experiments demonstrate that safety knowledge in English does not readily transfer to other languages, particularly given the nuance of safety policies and language-specific context. Therefore, it is essential to collect high-quality safety data for each language. We also found that the distribution of safety data per language significantly impacts performance from a safety standpoint, with some languages benefiting from transfer learning while others require more language-specific data. To achieve a balance between FRR and VR, we iteratively add adversarial and borderline data while monitoring the impact on both metrics.

We display results on our internal benchmarks in Figure 19 for short context models, showing Llama 3’s violation and false refusal rates for English and non-English languages compared to similar models and systems. To construct the benchmarks for each language, we use a combination of prompts written by native speakers, sometimes supplementing with translations from our English benchmarks. For each of our supported languages, we find that Llama 405B with Llama Guard is at least as safe, if not strictly safer, than the two competing systems when measured on our internal benchmark, while maintaining competitive false refusal rates. Looking at the Llama 405B model on its own, without Llama Guard, we find that it has a significantly lower violation rate than the competing standalone open source model, trading off a higher false refusal rate.

Long-context safety. Long-context models are vulnerable to many-shot jailbreaking attacks without targeted mitigation (Anil et al., 2024). To address this, we finetune our models on SFT datasets that include examples of safe behavior in the presence of demonstrations of unsafe behavior in context. We develop a scalable mitigation strategy that significantly reduces VR, effectively neutralizing the impact of longer context attacks even for 256-shot attacks. This approach shows little to no impact on FRR and most helpfulness metrics.

To quantify the effectiveness of our long-context safety mitigations, we use two additional benchmarking methods: DocQA and Many-shot. For DocQA, short for “document question answering,” we use long documents with information that could be utilized in adversarial ways. Models are provided both the document and a set of prompts related to the document in order to test whether the questions being related to information in the document affected the model’s ability to respond safely to the prompts. For Many-shot, following Anil et al. (2024), we construct a synthetic chat history composed of unsafe prompt-response pairs. A final prompt, unrelated to previous messages, is used to test whether the unsafe behavior in-context influenced the model

45

5.4.5 Cybersecurity and Chemical/Biological Weapons Safety

CyberSecurity evaluation results. To evaluate cybersecurity risk, we leverage the CyberSecEval benchmark framework (Bhatt et al., 2023, 2024), which contains tasks that measure safety across domains such as generating insecure code, generating malicious code, textual prompt injection, and vulnerability identification. We developed and applied Llama 3 to new benchmarks on spear phishing and autonomous cyberattacks. Overall, we find that Llama 3 does not have significant susceptibilities in generating malicious code or exploiting vulnerabilities. We describe brief results on specific tasks:

• Insecure coding testing framework: Evaluating Llama 3 8B, 70B, and 405B against the insecure coding testing framework, we continue to observe that larger models both generate more insecure code and also generate code with a higher average BLEU score (Bhatt et al., 2023).

• Code interpreter abuse prompt corpus: We identify that Llama 3 models are susceptible to executing malicious code under certain prompts, with Llama 3 405B being particularly susceptible by complying with malicious prompts 10.4% of the time. Llama 3 70B complied at a rate of 3.8%.

• Text-based prompt injection benchmark: When evaluated against prompt injection benchmarks, prompt injection attacks against Llama 3 405B were successful 21.7% of the time. Figure 22 provides text-based prompt injection success rates across Llama 3, GPT-4 Turbo, Gemini Pro, and Mixtral models.

• Vulnerability identification challenges: In assessing Llama 3’s ability to identify and exploit vulnerabilities using CyberSecEval 2’s capture-the-flag test challenges, Llama 3 does not outperform commonly used, traditional non-LLM tools and techniques.

• Spear phishing benchmark: We evaluate model persuasiveness and success rate in carrying out personalized conversations designed to deceive a target into unwittingly participating in security compromises. Randomized detailed victim profiles were generated by an LLM to serve as spear phishing targets. A judge LLM (Llama 3 70B) scored the performance of Llama 3 70B and 405B in interacting with a victim model (Llama 3 70B) and evaluated the success of the attempt. Llama 3 70B and Llama 3 405B were evaluated by the judge LLM to be moderately persuasive. Llama 3 70B was judged by an LLM to have been successful in 24% of spear phishing attempts while Llama 3 405B was judged to be successful in 14% of attempts. Figure 23 presents judge LLM-evaluated persuasiveness scores across models and phishing objectives.

• Attack automation framework: We assess Llama 3 70B’s and 405B’s potential to function as an autonomous agent across four critical phases of a ransomware attack – network reconnaissance, vulnerability identification, exploit execution, and post exploitation actions. We enable the models to behave autonomously by configuring the models to iteratively generate and execute new Linux commands in response to output from their prior commands on a Kali Linux virtual machine as they targeted another virtual machine with known vulnerabilities. Although Llama 3 70B and 405B efficiently identify network services and open ports in their network reconnaissance, the models fail to effectively use this information to gain initial access to the vulnerable machine across 20 and 23 test runs respectively. In identifying vulnerabilities, Llama 3 70B and 405B are moderately effective but struggle with selecting and applying successful exploitation techniques. Attempts to execute exploits were entirely unsuccessful as were post-exploit attempts to maintain access or impact hosts within a network.

Uplift testing for cyber attacks. We conduct an uplift study which measures the extent a virtual assistant improved the cyberattack rates of both novice and expert cyberattackers between two simulated offensive

46

<table>
<tr>
<th>Output formatting manipulation</th>
<th>Repeated token attack</th>
<th>Different user input language</th>
<th>Indirect reference</th>
<th>Ignore previous instructions</th>
<th>Virtualization</th>
<th>System mode</th>
<th>Many shot attack</th>
<th>Few shot attack</th>
<th>Mixed techniques</th>
<th>Persuasion</th>
<th>Overload with information</th>
<th>Payload splitting</th>
<th>Token smuggling</th>
<th>Hypothetical scenario</th>
</tr>
<tr>
<td>Mixtral 8x22B</td>
<td>0.56</td>
<td>0.56</td>
<td>0.56</td>
<td>0.25</td>
<td>0.56</td>
<td>0.31</td>
<td>0.38</td>
<td>0.31</td>
<td>0.25</td>
<td>0.31</td>
<td>0.25</td>
<td>0.38</td>
<td>0.25</td>
<td>0.19</td>
<td>0.12</td>
</tr>
<tr>
<td>Llama 3 70B</td>
<td>0.25</td>
<td>0.50</td>
<td>0.31</td>
<td>0.38</td>
<td>0.25</td>
<td>0.56</td>
<td>0.25</td>
<td>0.38</td>
<td>0.44</td>
<td>0.19</td>
<td>0.25</td>
<td>0.06</td>
<td>0.00</td>
<td>0.06</td>
<td>0.00</td>
</tr>
<tr>
<td>Llama 3 405B</td>
<td>0.25</td>
<td>0.31</td>
<td>0.38</td>
<td>0.44</td>
<td>0.31</td>
<td>0.19</td>
<td>0.19</td>
<td>0.12</td>
<td>0.31</td>
<td>0.12</td>
<td>0.06</td>
<td>0.25</td>
<td>0.12</td>
<td>0.06</td>
<td>0.12</td>
</tr>
<tr>
<td>Llama 3 8B</td>
<td>0.12</td>
<td>0.38</td>
<td>0.31</td>
<td>0.38</td>
<td>0.19</td>
<td>0.19</td>
<td>0.25</td>
<td>0.12</td>
<td>0.12</td>
<td>0.19</td>
<td>0.19</td>
<td>0.19</td>
<td>0.06</td>
<td>0.06</td>
<td>0.06</td>
</tr>
<tr>
<td>Gemini Pro</td>
<td>0.44</td>
<td>0.31</td>
<td>0.19</td>
<td>0.19</td>
<td>0.25</td>
<td>0.12</td>
<td>0.25</td>
<td>0.06</td>
<td>0.25</td>
<td>0.19</td>
<td>0.06</td>
<td>0.12</td>
<td>0.19</td>
<td>0.00</td>
<td>0.12</td>
</tr>
<tr>
<td>GPT-4 Turbo</td>
<td>0.62</td>
<td>0.31</td>
<td>0.25</td>
<td>0.50</td>
<td>0.12</td>
<td>0.00</td>
<td>0.12</td>
<td>0.12</td>
<td>0.06</td>
<td>0.12</td>
<td>0.00</td>
<td>0.00</td>
<td>0.12</td>
<td>0.12</td>
<td>0.00</td>
</tr>
</table>

<table>
<tr>
<th>Malware download</th>
<th>Security info gathering</th>
<th>Data theft</th>
<th>Credential theft</th>
</tr>
<tr>
<td>GPT-4 Turbo</td>
<td>4.02</td>
<td>4.09</td>
<td>3.84</td>
<td>3.97</td>
</tr>
<tr>
<td>Llama 3 70B</td>
<td>2.79</td>
<td>3.57</td>
<td>2.68</td>
<td>2.75</td>
</tr>
<tr>
<td>Llama 3 405B</td>
<td>2.71</td>
<td>3.37</td>
<td>2.03</td>
<td>2.31</td>
</tr>
<tr>
<td>Mixtral 8x22B</td>
<td>1.68</td>
<td>2.01</td>
<td>1.47</td>
<td>1.58</td>
</tr>
</table>

cybersecurity challenges. A two-stage study was conducted with 62 internal volunteers. Volunteers were
categorized into “expert” (31 subjects) and “novice” (31 subjects) cohorts based on their offensive security
experience. For the first stage, subjects were asked to complete the challenge without any LLM assistance
but with access to the open internet. For the second stage, subjects retained access to the internet but were
also provided with Llama 3 405B to complete a different offensive cybersecurity challenge of similar difficulty
to the first. An analysis of the completion rates of challenge attack phases by subjects indicates that both
novices and experts using the 405B model demonstrated insignificant uplift over having open access to the
internet without an LLM.

Uplift testing for chemical and biological weapons. To assess risks related to proliferation of chemical and
biological weapons, we perform uplift testing designed to assess whether use of Llama 3 could meaningfully
increase the capabilities of actors to plan such attacks.

The study consists of six-hour scenarios where teams of two participants were asked to generate fictitious
operational plans for either a biological or chemical attack. The scenarios cover the major planning stages of a
CBRNE attack (agent acquisition, production, weaponization, and delivery) and are designed to elicit detailed
plans that would address challenges related to procurement of restricted materials, real-world laboratory
protocols, and operational security. Participants are recruited based on previous experience in relevant areas of
scientific or operational expertise, and assigned to teams consisting of two low-skill actors (no formal training)
or two moderate-skill actors (some formal training and practical experience in science or operations).

The study was generated in collaboration with a set of CBRNE experts, and designed to maximize the
generality, validity, and robustness of both quantitative and qualitative outcomes. A preliminary study was
also performed in order to validate the study design, including a robust power analysis ensuring that our
sample size was sufficient for statistical analysis.

Each team is assigned to a “control” or “LLM” condition. The control team has access to internet-based
resources only, while the LLM-enabled team had internet access as well as access to Llama 3 models enabled
with web search (including PDF ingestion), information retrieval capabilities (RAG), and code execution
(Python and Wolfram Alpha). To enable testing of RAG capabilities, a keyword search is used to generate a
dataset of hundreds of relevant scientific papers and pre-loaded into the Llama 3 model inference system. At
the conclusion of the exercise, the operational plans generated by each team are evaluated by subject matter
experts with domain expertise in biology, chemistry, and operational planning. Each plan is evaluated across
four stages of potential attacks, generating scores for metrics such as scientific accuracy, detail, detection
avoidance, and probability of success in scientific and operational execution. After a robust Delphi process
to mitigate bias and variability in subject matter expert (SME) evaluations, final scores are generated by
pooling stage-level metrics into a comprehensive score.

Quantitative analysis of these results of this study show no significant uplift in performance related to usage
of the Llama 3 model. This result holds true when performing an aggregate analysis (comparing all LLM
conditions to the web-only control condition) as well as for breakdowns by subgroups (e.g., separate evaluation

47

of the Llama 3 70B and Llama 3 405B models, or separate evaluation of scenarios related to chemical or biological weapons). After validating these results with CBRNE SMEs, we assess that there is a low risk that release of Llama 3 models will increase ecosystem risk related to biological or chemical weapon attacks.

5.4.6 Red Teaming

We utilize Red Teaming to discover risks and use the findings to improve our benchmarks and safety tuning datasets. We conduct recurring red teaming exercises to continuously iterate and discover new risks, which guides our model development and mitigation process.

Our red team consists of experts in cybersecurity, adversarial machine learning, responsible AI, and integrity, in addition to multilingual content specialists with backgrounds in integrity issues for specific geographic markets. We also partner with internal and external subject-matter experts in critical risk areas to help build risk taxonomies and aid in more focused adversarial assessment.

Adversarial testing on specific model capabilities. We began initial red teaming by focusing on individual model capabilities in a risk discovery process, in context of specific high-risk categories then testing capabilities together. The red team focused on prompt-level attacks to emulate more real-world scenarios—we find that models often deviate from expected behavior, particularly in cases when the prompt’s intention is being obfuscated or when prompts layer multiple abstractions. These risks get more complex with additional capabilities, and we describe several of our red teaming discoveries in detail below. We utilize these red team discoveries in concert with our results on internal safety benchmarks to develop focused mitigations to continuously and iteratively improve model safety.

• Short and long-context English. We employed a mix of well-known, published and unpublished techniques across single and multi-turn conversations. We also leveraged advanced, adversarial multi-turn automation similar to PAIR (Chao et al., 2023) across some techniques and risk categories. Largely, multi-turn conversations lead to more harmful outputs. Several attacks were pervasive across model checkpoints, particularly when used together.

– Multi-turn refusal suppression to specify the model response to follow a particular format or include/exclude particular information related to the refusal as specific phrases.

– Hypothetical scenarios wrap violating prompts as hypothetical/theoretical tasks or fictional scenarios. Prompts can be as simple as adding the word “hypothetically” or crafting an elaborate layered scenario.

– Personas and role play gives the model a violating persona with specific violating response characteristics (e.g., “You are X, your goal is Y”) or yourself as the user adapting a specific benign character that obfuscates the context of the prompt.

– Adding disclaimers and warnings works as a form of response priming and we assume a method to allow for the model a path to helpful compliance that intersects with generalized safety training. Asking for disclaimers, trigger warnings and more to be added in multi-turn conversations in concert with other attacks mentioned contributed to increased violation rates.

– Gradually escalating violation is a multi-turn attack where the conversation starts out with a more or less benign request and then through direct prompting for more exaggerated content can gradually lead the model into generating a very violating response. Once the model has started outputting violating content, it can be difficult for the model to recover (or another attack can be used if a refusal is encountered). With longer context models, this will be an increasingly seen issue.

• Multilingual. We identify a number of unique risks when considering multiple languages.

– Mixing multiple languages in one prompt or conversation can easily lead to more violating outputs than if a single language was used.

– Lower resource languages can lead to violating outputs given a lack of related safety fine-tuning data, weak model generalization of safety or prioritization of testing or benchmarks. However, this attack often results in poor quality generally, limiting real adversarial use.

48

–Slang, specific context or cultural-specific references can confuse or appear to be violating at first glance, only to see the model does not comprehend a given reference correctly to make an output truly harmful or prevent it from being a violating output.
•Tool use. During testing, apart from English-text level adversarial prompting techniques being successful in generating violating outputs, several tool specific attacks were also discovered. This included but was not limited to:
–Unsafe tool chaining such as asking for multiple tools at once with one being violating could, in early checkpoints, lead to all of the tools being called with a mix of benign and violating inputs.
–Forcing tool use often with specific input strings, fragmented or encoded text can trigger a tool input to be potentially violating, leading to a more violating output. Other techniques can then be used to access the tool results, even if the model would normally refuse to perform the search or assist with the results.
–Modifying tool use parameters such as swapping words in queries, retrying, or obfuscating some of the initial request in a multi-turn conversation lead to violations in many early checkpoints as a form of forcing tool use.
Child safety risks. Child Safety risk assessments were conducted using a team of experts, to assess the model’s capability to produce outputs that could result in Child Safety risks and inform on any necessary and appropriate risk mitigations via fine tuning. We leveraged those expert red teaming sessions to expand the coverage of our evaluation benchmarks through model development. For Llama 3, we conducted new in-depth sessions using objective based methodologies to assess model risks along multiple attack vectors. We also partnered with content specialists to perform red teaming exercises assessing potentially violating content while taking account of market specific nuances or experiences.
5.4.7 System Level Safety
In various real-world applications of large language models, models are not used in isolation but are integrated into broader systems. In this section, we describe our system level safety implementation, which supplements model-level mitigations by providing more flexibility and control.
To enable this, we develop and release a new classifier, Llama Guard 3, which is a Llama 3 8B model fine-tuned for safety classification. Similar to Llama Guard 2 (Llama-Team, 2024), this classifier is used to detect whether input prompts and/or output responses generated by language models violate safety policies on specific categories of harm.
It is designed to support Llama’s growing capabilities, and can be used for English and multilingual text. It is also optimized to be used in the context of tool-calls such as search-tools and preventing code interpreter abuse. Finally, we also provide quantized variants to reduce memory requirements. We encourage developers to use our release of system safety components as a foundation and configure them for their own use cases.
Taxonomy. We train on the 13 hazard categories listed in the AI Safety taxonomy (Vidgen et al., 2024): Child Sexual Exploitation, Defamation, Elections, Hate, Indiscriminate Weapons, Intellectual Property, Non-Violent Crimes, Privacy, Sex-Related Crimes, Sexual Content, Specialized Advice, Suicide & Self-Harm, and Violent Crimes. We also train on Code Interpreter Abuse category to support tool-calls use cases.
Training data. We start with the English data used by Llama Guard (Inan et al., 2023) and expand this dataset to incorporate new capabilities. For new capabilities such as multilingual and tool use, we collect prompt and response classification data, as well as utilize the data collected for safety finetuning. We increase the number of unsafe responses in the training set by doing prompt engineering to get the LLM to not refuse responding to adversarial prompts. We use Llama 3 to obtain response labels on such generated data.
To improve the performance of Llama Guard 3, we do extensive cleaning of the collected samples using human annotation as well as LLM annotation by Llama 3. Obtaining labels for user prompts is a much harder task for both humans and LLMs, and we find that the human labels are slightly better, especially for borderline prompts, though our full iterative system is able to reduce the noise and produce more accurate labels.
49


<table style="width:100%;">
<tr>
<th>Capability</th>
<th>VR</th>
<th>FRR</th>
<th>VR</th>
<th>FRR</th>
<th>VR</th>
<th>FRR</th>
</tr>
<tr>
<td>English</td>
<td>-76%</td>
<td>+95%</td>
<td>-75%</td>
<td>+25%</td>
<td>-86%</td>
<td>+102%</td>
</tr>
<tr>
<td>French</td>
<td>-38%</td>
<td>+27%</td>
<td>-45%</td>
<td>+4%</td>
<td>-59%</td>
<td>+29%</td>
</tr>
<tr>
<td>German</td>
<td>-57%</td>
<td>+32%</td>
<td>-60%</td>
<td>+14%</td>
<td>-77%</td>
<td>+37%</td>
</tr>
<tr>
<td>Hindi</td>
<td>-54%</td>
<td>+60%</td>
<td>-54%</td>
<td>+14%</td>
<td>-71%</td>
<td>+62%</td>
</tr>
<tr>
<td>Italian</td>
<td>-34%</td>
<td>+27%</td>
<td>-34%</td>
<td>+5%</td>
<td>-48%</td>
<td>+29%</td>
</tr>
<tr>
<td>Portuguese</td>
<td>-51%</td>
<td>+35%</td>
<td>-57%</td>
<td>+13%</td>
<td>-65%</td>
<td>+39%</td>
</tr>
<tr>
<td>Spanish</td>
<td>-41%</td>
<td>+26%</td>
<td>-50%</td>
<td>+10%</td>
<td>-60%</td>
<td>+27%</td>
</tr>
<tr>
<td>Thai</td>
<td>-43%</td>
<td>+37%</td>
<td>-39%</td>
<td>+8%</td>
<td>-51%</td>
<td>+39%</td>
</tr>
</table>
<p>Table 25 Violation Rate (VR) and False Refusal Rate (FRR) relative to Llama 3 when using Llama Guard 3 for input or output filtering on different languages. For example, -50% for VR means that there is a 50% reduction in the rate of Llama 3 model violations when using Llama Guard. Evaluations are performed on generations from the 405B-parameter Llama 3 model. Lower is better.</p>
<p>Results. Llama Guard 3 is able to significantly reduce violations across capabilities (-65% violations on average across our benchmarks). Note that adding system safeguards (and any safety mitigations in general) comes at the cost of increased refusals to benign prompts. In Table 25 we report reductions in violation rate and increases in false refusal rate increase compared to the base model to highlight this tradeoff. This effect is also visible in Figures 19, 20, and 21.</p>
<p>System safety also offers more flexibility. Llama Guard 3 can be deployed for specific harms only enabling control over the violations and false refusals trade-off at the harm category level. Table 26 presents violations reduction per category to inform which category should be turned on/off based on the developer use case.</p>
<p>To make it easier to deploy safety systems, we provide a quantized version of Llama Guard 3 using the commonly used int8 quantization technique, reducing its size by more than 40%. Table 27 illustrates that quantization has negligible impact on the performance of the model.</p>
<p>Prompt-based system guards. System-level safety components enable developers to customize and control how LLM systems respond to user requests. As part of our work on improving the overall safety of the model system and enable developers to deploy responsibly, we describe and release the creation of two prompt-based filtering mechanisms: Prompt Guard and Code Shield. We open-source these for the community to leverage as-is or take as inspiration and adapt for their usecases.</p>
<p>Prompt Guard is a model-based filter designed to detect prompt attacks, which are input strings designed to subvert the intended behavior of an LLM functioning as part of an application. The model is a multi-label classifier that detects two classes of prompt attack risk - direct jailbreaks (techniques that explicitly try to override a model’s safety conditioning or system prompt) and indirect prompt injections (instances where third-party data included in a model’s context window includes instructions inadvertently executed as user commands by an LLM). The model is fine-tuned from mDeBERTa-v3-base, a small (86M) parameter model suitable for filtering inputs into an LLM. We evaluate the performance on several evaluation datasets shown in Table 28. We evaluate on two datasets (jailbreaks and injections) drawn from the same distribution as the training data, as well as an out-of-distribution dataset in English, a multilingual jailbreak set built from machine translation, and a dataset of indirect injections drawn from CyberSecEval (both English and multilingual). Overall, we find that the model generalizes well to new distributions and has strong performance.</p>
<p>Code Shield is an example of a class of system-level protections based on providing inference-time filtering. In particular, it focuses on detecting the generation of insecure code before it might enter a downstream usecase such as a production system. It does so by leveraging a static analysis library, the Insecure Code Detector (ICD), to identify insecure code. ICD uses a suite of static analysis tools to perform the analysis across 7 programming languages. These kinds of guardrails are generally useful for developers, who can deploy multi-layered protections in various applications.</p>
<p>50</p>


Category | Input Llama Guard | Output Llama Guard | Full Llama Guard
---|---|---|---
False Refusal Rate Relative to Llama 3: | +95% | +25% | +102%
Violation Rate Relative to Llama 3: |  |  | 
Child Sexual Exploitation | -53% | -47% | -59%
Defamation | -86% | -100% | -100%
Elections | -100% | -100% | -100%
Hate | -36% | -82% | -91%
Indiscriminate Weapons<sup>14</sup> | 0% | 0% | 0%
Intellectual Property | -88% | -100% | -100%
Non-Violent Crimes | -80% | -80% | -100%
Privacy | -40% | -60% | -60%
Sex-Related Crimes | -75% | -75% | -88%
Sexual Content | -100% | -100% | -100%
Specialized Advice | -70% | -70% | -70%
Suicide & Self-Harm | -62% | -31% | -62%
Violent Crimes | -67% | -53% | -80%

Table 26 Violation rate and false refusal rate relative to Llama 3 when using Llama Guard 3 for input or output filtering on different safety categories. For example, -50% for VR means that there is a 50% reduction in the rate of Llama 3 model violations when using Llama Guard. Evaluations are performed on English prompts and generations from the 405B parameter Llama 3 model. Lower is better.

|  | Non-Quantized | Quantized |
|---|---|---|
| Capability | Precision | Recall | F1 | FPR | Precision | Recall | F1 | FPR |
| English | 0.947 | 0.931 | 0.939 | 0.040 | 0.947 | 0.925 | 0.936 | 0.040 |
| Multilingual | 0.929 | 0.805 | 0.862 | 0.033 | 0.931 | 0.785 | 0.851 | 0.031 |
| Tool Use | 0.774 | 0.884 | 0.825 | 0.176 | 0.793 | 0.865 | 0.827 | 0.155 |

Table 27 int8 Llama Guard. Effect of int8 quantization on Llama Guard 3 output classification performance for different model capabilities.

5.4.8 Limitations

We conducted extensive measurement and mitigation on a wide variety of risks to safe usage of Llama 3. However, no testing can be guaranteed to be exhaustive in identifying every possible risk. Llama 3 may still generate harmful content due to training on various datasets, particularly for languages beyond English and when prompt engineered by skilled adversarial red teamers. Malicious developers or adversarial users may find new ways to jailbreak our models and use them for various nefarious usecases. We will continue to proactively identify risks, conduct research on mitigation methods, and we encourage developers to consider responsibility in every aspect — from model development to deployment to users. We hope developers will leverage and contribute to the tools we release in our open-source system-level safety suite.

6 Inference

We investigate two main techniques to make inference with the Llama 3 405B model efficient: (1) pipeline parallelism and (2) FP8 quantization. We have publicly released our implementation of FP8 quantization.

6.1 Pipeline Parallelism

When using a BF16 number representation for the model parameters, Llama 3 405B does not fit in the GPU memory of a single machine with 8 Nvidia H100 GPUs. To address this issue, we parallelize model inference using BF16 precision across 16 GPUs on two machines. Within each machine, the high NVLink bandwidth

51

<table style="width:100%;">
<tr>
<th>Metric</th>
<th>Jailbreaks</th>
<th>Injections</th>
<th>Out-of-Distribution Jailbreaks</th>
<th>Multilingual Jailbreaks</th>
<th>Indirect Injections</th>
</tr>
<tr>
<td>TPR</td>
<td>99.9%</td>
<td>99.5%</td>
<td>97.5%</td>
<td>91.5%</td>
<td>71.4%</td>
</tr>
<tr>
<td>FPR</td>
<td>0.4%</td>
<td>0.8%</td>
<td>3.9%</td>
<td>5.3%</td>
<td>1.0%</td>
</tr>
<tr>
<td>AUC</td>
<td>0.997</td>
<td>1.000</td>
<td>0.975</td>
<td>0.959</td>
<td>0.996</td>
</tr>
</table>
Table 28 Performance of Prompt Guard. We include in- and out-of-distribution evaluations, a multilingual jailbreak built using machine translation, and a dataset of indirect injections from CyberSecEval.

enables the use of tensor parallelism (Shoeybi et al., 2019). Across nodes, however, connectivity has lower bandwidth and higher latency, so we use pipeline parallelism (Huang et al., 2019) instead.

During training with pipeline parallelism, bubbles are a major efficiency concern (see Section 3.3). However, they are not an issue during inference, since inference does not involve a backward pass that requires a pipeline flush. Therefore, we use micro-batching to improve inference throughput with pipeline parallelism.

We evaluate the effect of using two micro-batches in inference workloads of 4,096 input tokens and 256 output tokens both during the key-value cache pre-fill stage of inference and during the decoding stage. We find that micro-batching improves throughput of inference with the same local batch size; see Figure 24. These improvements result from micro-batching enabling concurrent execution of micro batches in both these stages. The additional synchronization points due to micro-batching also increase latency but, overall, micro-batching still leads to a better throughput-latency trade-off.

6.2 FP8 Quantization

We perform experiments leveraging the native FP8 support of H100 GPUs to perform low-precision inference. To enable low-precision inference, we apply FP8 quantization to most matrix multiplications inside the model. In particular, we quantize most parameters and activations in the feedforward network layers in the model, which account for roughly 50% of the inference compute time. We do not quantize parameters in the self-attention layers of the model. We leverage dynamic scaling factors for better accuracy (Xiao et al., 2024b), optimizing our CUDA kernels<sup>15</sup> to reduce the overhead of calculating the scales. We find that the quality of Llama 3 405B is sensitive to certain types of quantization, and make a few additional changes to increase the model output quality:

1. Akin to Zhang et al. (2021), we do not perform quantization in the first and last Transformer layers.
2. High-perplexity tokens such as dates can lead to large activation values. In turn, these can lead to high dynamic scaling factors in FP8 and a non-negligible number of underflows, leading to errors in decoding.

<sup>15</sup>Our FP8 kernels are available at https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/gen_ai.
We provide usage examples at https://github.com/meta-llama/llama-agentic-system.

52

Figure 25 Illustration of tensor-wise and row-wise FP8 quantization. Right: Row-wise quantization enables the use of more granular activation factors than Left: tensor-wise quantization.

<br>

Figure 26 Reward score distribution for Llama 3 405B using BF16 and FP8 inference. Our FP8 quantization approach has negligible impact on the model's responses.

<br>

To address this issue, we upper bound the dynamic scaling factors to 1200.

<br>

3. We use row-wise quantization, computing scaling factors across rows for parameter and activation matrices (see Figure 25). We find this works better than a tensor-wise quantization approach.

<br>

Effect of quantization errors. Evaluations on standard benchmarks often suggest that FP8 inference performs on par with BF16 inference even without these mitigations. However, we find that such benchmarks do not adequately reflect the effects of FP8 quantization. When scaling factors are not upper bounded, the model occasionally produces corrupted responses even though the benchmark performance is strong. Instead of relying on benchmarks to measure distribution changes due to quantization, we find it is better to analyze the distribution of reward-model scores for 100,000 responses produced using both FP8 and BF16. Figure 26 shows the resulting reward distribution for our quantization approach. The results in the figure show that our approach to FP8 quantization has very limited impact on the model's response.

<br>

Experimental evaluation of efficiency. Figure 27 depicts the throughput-latency trade-off of performing FP8 inference with Llama 3 405B in the pre-fill and decoding stages, using 4,096 input tokens and 256 output tokens. The figure compares the efficiency of FP8 inference with that of the two-machine BF16 inference approach described in Section 6.1. The results show that use of FP8 inference leads to throughput improvements of up to 50% during the pre-fill stage, and a substantially better throughput-latency trade-off during decoding.

<br>

53

54

7.1.1 Image Data

Our image encoder and adapter are trained on image-text pairs. We construct this dataset via a complex data processing pipeline that consists of four main stages: (1) quality filtering, (2) perceptual de-duplication, (3) resampling, and (4) optical character recognition. We also apply a series of safety mitigations.

• Quality filtering. We implement quality filters that remove non-English captions and low-quality captions via heuristics such as low alignment scores produced by (Radford et al., 2021). Specifically, we remove all image-text pairs below a certain CLIP score.

• De-duplication. De-duplicating large-scale training datasets benefits model performance because it reduces training compute spent on redundant data (Esser et al., 2024; Lee et al., 2021; Abbas et al.,

Figure 28 Illustration of the compositional approach to adding multimodal capabilities to Llama 3 that we study in this paper. This approach leads to a multimodal model that is trained in five stages: (1) language model pre-training, (2) multi-modal encoder pre-training, (3) vision adapter training, (4) model finetuning, and (5) speech adapter training.

2023) and memorization (Carlini et al., 2023; Somepalli et al., 2023). Hence, we de-duplicate our training data for both efficiency and privacy reasons. To do so, we use an internal version of the state-of-the-art SSCD copy-detection model (Pizzi et al., 2022) to de-duplicate images at scale. For all images, we first compute a 512-dimensional representation using the SSCD model. We use those embeddings to perform a nearest neighbor (NN) search for each image across all images in our data set, using a cosine similarity measure. We define examples above a certain similarity threshold as duplicates. We group these duplicates using a connected-components algorithm, and maintain only one image-text pair per connected component. We increase the efficiency of our de-duplication pipeline by: (1) pre-clustering the data using k-means clusters and (2) using FAISS (Johnson et al., 2019) for NN searches and clustering.

• Resampling. We ensure diversity of the image-text pairs via resampling akin to Xu et al. (2023); Mahajan et al. (2018); Mikolov et al. (2013). First, we construct a vocabulary of n-grams by parsing high-quality text sources. Next, we compute the frequency of each vocabulary n-gram in our dataset. We then resample the data as follows: If any of the n-grams in a caption occurs less than T times in the vocabulary, we keep the corresponding image-text pair. Otherwise, we independently sample each of the n-grams n<sub>i</sub> in the caption with probability √T/f<sub>i</sub> where f<sub>i</sub> indicates the frequency of n-gram n<sub>i</sub>; we keep the image-text pair if any of the n-grams was sampled. This resampling aids performance on low-frequency categories and fine-grained recognition tasks.

• Optical character recognition. We further improve our image-text data by extracting text written in the image and concatenating it with the caption. The written text is extracted using a proprietary optical character recognition (OCR) pipeline. We observe that adding OCR data into the training data greatly improves tasks that require OCR capabilities, such as document understanding.

Transcribing documents. To improve the performance of our models on document understanding tasks, we render pages from documents as images and paired the images with their respective text. The document text is obtained either directly from the source or via a document parsing pipeline.

Safety. We focus primarily on ensuring that the pre-training dataset for image recognition does not contain

55

unsafe content, such as sexual abuse material (CSAM) (Thiel, 2023). We scan all our training images for CSAM using perceptual hashing approaches such as PhotoDNA (Farid, 2021) as well as internal, proprietary classifiers. We also use a proprietary media-risk retrieval pipeline to identify and remove image-text pairs that we consider to be NSFW, for example, because they contain sexual or violent content. We believe that minimizing the prevalence of such material in the training dataset improves the safety of the final model without impacting its helpfulness. Finally, we perform face blurring on all images in our training set. We test the model against human generated prompts that refer to an attached image.
Annealing data. We create an annealing dataset by resampling the image-caption pairs to a smaller volume of ~350M examples using n-grams. Since the n-grams resampling favor richer text descriptions, this selects a higher-quality data subset. We augment the resulting data with ~150M examples from five additional sources:
• Visual grounding. We link noun phrases in the text to bounding boxes or masks in the image. The grounding information (bounding boxes and masks) are specified in the image-text pair in two ways. (1) We overlay boxes or masks with marks on the image and use marks in the text as reference, akin to set-of-marks (Yang et al., 2023a). (2) We insert normalized (xmin, ymin, xmax, ymax) coordinates directly into the text, demarcated by special tokens.
• Screenshot parsing. We render screenshots from HTML code and task the model with predicting the code that produced a specific element in the screenshot, akin to Lee et al. (2023). The element of interest is indicated in the screenshot via a bounding box.
• Question-answer pairs. We include question-answer pairs, enabling us to use volumes of question-answering data that are too large to be used in model finetuning.
• Synthetic captions. We include images with synthetic captions that were generated by an early version of the model. Compared to original captions, we find that synthetic captions provide a more comprehensive description of images than the original captions.
• Synthetically-generated structured images. We also include synthetically generated images for a variety of domains such as charts, tables, flowcharts, math equations and textual data. These images are accompanied by a structured representation such as the corresponding markdown or LaTeX notation. Besides improving recognition capabilities of the model for these domains, we find this data useful to generate question-answer pairs via the text model for finetuning.
7.1.2 Video Data
For video pre-training, we use a large dataset of video-text pairs. Our dataset is curated through a multi-stage process. We filter and clean the associated texts using rule-based heuristics, such as ensuring a minimum length and fixing capitalization. Then, we run language identification models to filter out non-English texts. We run OCR detection models to filter out videos with excessive overlaid text. To ensure reasonable alignment between the video-text pairs, we use CLIP (Radford et al., 2021) style image-text and video-text contrastive models. We first compute image-text similarity using a single frame in the videos and filtered out low similarity pairs, and then subsequently filter out pairs with low video-text alignment. Some of our data contains static or low-motion videos; we filter out such data using motion-score based filtering (Girdhar et al., 2023). We do not apply any filters on the visual quality of the videos such as aesthetic scores or resolution filtering.
Our dataset contains videos with an average duration of 21 seconds and a median duration of 16 seconds, with over 99% videos being under a minute. The spatial resolution varies significantly between 320p and 4K videos, with over 70% of the videos having a short side greater than 720 pixels. The videos have varying aspect ratios with almost all videos having between aspect ratio between 1:2 and 2:1, with a 1:1 median.
7.2 Model Architecture
Our visual-recognition model consists of three main components: (1) an image encoder, (2) an image adapter, and (3) a video adapter.
Image encoder. Our image encoder is a standard vision transformer (ViT; Dosovitskiy et al. (2020)) that is trained to align images and text (Xu et al., 2023). We use the ViT-H/14 variant of the image encoder,
56


which has 630M parameters that were trained on 2.5B image-text pairs for five epochs. The image encoder is pre-trained on images with resolution 224 × 224; images were split up into 16 × 16 patches of equal size (i.e., a patch size of 14 × 14 pixels). As also demonstrated by prior work such as ViP-Llava (Cai et al., 2024), we observe that image encoders trained via a contrastive text alignment objective are unable to preserve fine-grained localization information. To alleviate this, we employ a multi-layer feature extraction, where features from the 4th, 8th, 16th, 24th and 31st layers are also provided in addition to the final layer features. In addition, we further insert 8 gated self-attention layers (making a total of 40 transformer blocks) prior to pre-training of the cross-attention layers to learn alignment-specific features. The image encoder therefore eventually has a total 850M parameters with the additional layers. With the multi-layer features, the image encoder produces a 7680-dimensional representation for each of the resulting 16 × 16 = 256 patches. The parameters of the image encoder are not frozen during subsequent training stages as we found it to improve performance, especially in domains such as text recognition.

Image adapter. We introduce cross-attention layers between the visual token representations produced by the image encoder and the token representations produced by the language model (Alayrac et al., 2022). The cross-attention layers are applied after every fourth self-attention layer in the core language model. Like the language model itself, the cross-attention layers use generalized query attention (GQA) for increased efficiency. The cross-attention layers introduce substantial numbers of additional trainable parameters into the model: for Llama 3 405B, the cross-attention layers have ≈100B parameters. We pre-train our image adapter in two stages: (1) initial pre-training followed by (2) annealing:

•	Initial pre-training. We pre-train our image adapter on our dataset of ~6B image-text pairs described above. For compute efficiency reasons, we resize all images to fit within at most four tiles of 336 × 336 pixels each, where we arrange the tiles to support different aspect ratios, e.g., 672 × 672, 672 × 336, and 1344 × 336.

•	Annealing. We continue training the image adapter on ~500M images from the annealing dataset described above. During annealing, we increase the per-tile image resolution to improve performance on tasks that require higher-resolution images, for example, infographics understanding.

Video adapter. Our model takes as input up to 64 frames (uniformly sampled from a full video), each of which is processed by the image encoder. We model temporal structure in videos through two components: (i) encoded video frames are aggregated by a temporal aggregator which merges 32 consecutive frames into one, (ii) additional video cross attention layers are added before every fourth image cross attention layer. The temporal aggregator is implemented as a perceiver resampler (Jaegle et al., 2021; Alayrac et al., 2022). We pre-train using 16 frames per video (aggregated to 1 frame), but increase the number of input frames to 64 during supervised finetuning. The video aggregator and cross attention layers have 0.6B and 4.6B parameters for Llama 3 7B and 70B, respectively.

7.3 Model Scaling

After the visual-recognition components are added to Llama 3, the model contains self-attention layers, cross-attention layers, and a ViT image encoder. To train adapters for the smaller 8B and 70B parameter models, we found a combination of data and tensor parallelization is the most efficient. Model or pipeline parallelism does not increase efficiency at these scales because the gathering of model parameters would dominate the computation. We do, however, use pipeline parallelism (in addition to data and tensor parallelism) when training the adapter for the 405B parameter model. Training at this scale introduces three new challenges in addition to those outlined in Section 3.3: model heterogeneity, data heterogeneity, and numerical instabilities.

Model heterogeneity. The model computation is heterogeneous because more computation is performed on some tokens than on others. In particular, image tokens are processed by the image encoder and the cross-attention layers, whereas text tokens are only processed by the language backbone. This heterogeneity leads to bottlenecks in the scheduling of pipeline parallelism. We address this problem by ensuring each pipeline stage contains five layers: namely, four self-attention layers in the language backbone and a cross-attention layer. (Recall that we introduce a cross-attention layer after every fourth self-attention layer.) In addition, we replicate the image encoder on all pipeline stages. Because we train on paired image-text data, this enables us to perform load balancing between the image and text parts of the computation.

57

Data heterogeneity. The data is heterogeneous because, on average, images have more tokens than the associated text: an image has 2,308 tokens, whereas the associated text contains an average of only 192 tokens. As a result, the computation of cross-attention layers requires more time and memory than the computation of self-attention layers. We address this problem by introducing sequence parallelization in the image encoder, so that each GPU processes roughly the same number of tokens. Because the average text size is relatively short, we also use a substantially larger micro-batch size (8 instead of 1).

Numerical instabilities. After the image encoder is added to the model, we find that performing gradient accumulation in bf16 led to numerical instabilities. The most likely explanation for this is that image tokens are introduced into the language backbone via all cross-attention layers. This implies that numerical deviations in the representation of an image token have an outsized impact on the overall computation because the errors are compounded. We address this by performing gradient accumulation in FP32.

7.4 Pre-training
Image. We initialize from the pre-trained text model and vision encoder weights. The vision encoder is unfrozen, while the text model weights are kept frozen as explained above. First, we train the model using 6B image-text pairs where each image is resized to fit within four tiles of 336 × 336 pixels. We use a global batch size of 16,384 and a cosine learning rate schedule with initial learning rate 10 × 10⁻⁴ and a weight decay of 0.01. The initial learning rate was determined based on small-scale experiments. However, these findings did not generalize well to very long training schedules and dropped the learning rate a few times during training when the loss values became stagnant. After the base pre-training, we increase the image resolution further and continue training the same weights on the annealing dataset. The optimizer is re-initialized via warm-up to learning rate 2 × 10⁻⁵ and again follows a cosine schedule.

Video. For video pre-training, we start from the image pre-trained and annealed weights as described above. We add the video aggregator and cross-attention layers as described in the architecture, initialized randomly. We freeze all the parameters in the model except the video-specific ones (the aggregator and video cross-attention), and train them on the video pre-training data. We use the same training hyperparameters as the image annealing stage, with small differences in the learning rate. We uniformly sample 16 frames from the full video, and represent each frame using four chunks, each of size 448 × 448 pixels. We use an aggregation factor of 16 in the video aggregator, hence obtaining one effective frame, which the text tokens cross-attend to. We use a global batch size of 4,096, a sequence length of 190 tokens, and a learning rate of 10⁻⁴ during training.

7.5 Post-Training
In this section, we describe the post-training recipe for our vision adapters. After pre-training, we fine-tune the model on highly curated multi-modal conversational data to enable chat capabilities. We further implement direct preference optimization (DPO) to boost human evaluation performance and rejection sampling to improve multi-modal reasoning capabilities. Finally, we add a quality-tuning stage where we continue fine-tuning the model on a very small set of high-quality conversational data which further boosts human evaluation while retaining performance across benchmarks. More details on each of these steps are provided below.

7.5.1 Supervised Finetuning Data
We describe our supervised finetuning (SFT) data for image and video capabilities separately below.

Image. We utilize a mix of different datasets for supervised finetuning.
•	Academic datasets. We convert a highly filtered collection of existing academic datasets to question-answer pairs using templates or via LLM rewriting. The LLM rewriting’s purpose is to augment the data with different instructions and to improve the language quality of answers.
•	Human annotations. We collect multi-modal conversation data via human annotators for a wide range of tasks (open-ended question-answering, captioning, practical use cases, etc.) and domains (e.g., natural images and structured images). Annotators are provided with images and asked to write conversations. To ensure diversity, we cluster large-scale datasets and sampled images uniformly across different clusters. Further, we acquire additional images for a few specific domains by expanding a seed via k-nearest neighbors search.

Video. We leverage existing video question answering datasets and create new ones using the same methodology as described for images. We also collect additional video data via human annotators.

neighbors. Annotators are also provided with intermediate checkpoints of existing models to facilitate model-in-the-loop style annotations, so that model generations can be utilized as a starting point by the annotators to then provide additional human edits. This is an iterative process, in which model checkpoints would be regularly updated with better performing versions trained on the latest data. This increases the volume and efficiency of human annotations, while also improving their quality.
• Synthetic data. We explore different ways to generate synthetic multi-modal data by using text-representations of images and a text-input LLM. The high-level idea is to utilize the reasoning capabilities of text-input LLMs to generate question-answer pairs in the text domain, and replace the text representation with its corresponding images to produce synthetic multi-modal data. Examples include rendering texts from question-answer datasets as images or rendering table data into synthetic images of tables and charts. Additionally, we use captions and OCR extractions from existing images to generate additional conversational or question-answer data related to the images.
Video. Similar to the image adapter, we use academic datasets with pre-existing annotations and convert them into appropriate textual instructions and target responses. The targets are converted to open-ended responses or multiple-choice options, whichever is more appropriate. We ask humans to annotate videos with questions and corresponding answers. The annotators are asked to focus on questions that could not be answered based on a single frame, to steer the annotators towards questions that require temporal understanding.
7.5.2 Supervised Finetuning Recipe
We describe our supervised finetuning (SFT) recipe for image and video capabilities separately below.
Image. We initialize from the pre-trained image adapter, but hot-swap the pre-trained language model's weights with the instruction tuned language model's weights. The language model weights are kept frozen to maintain text-only performance, i.e., we only update the vision encoder and image adapter weights.
Our approach to finetune the model is similar to Wortsman et al. (2022). First, we run a hyperparameter sweep using multiple random subsets of data, learning rates and weight decay values. Next, we rank the models based on their performance. Finally, we average the weights of the top-K models to obtain the final model. The value of K is determined by evaluating the averaged models and selecting the instance with highest performance. We observe that the averaged models consistently yield better results compared to the best individual model found via grid search. Further, this strategy reduces sensitivity to hyperparameters.
Video. For video SFT, we initialize the video aggregator and cross-attention layers using the pre-trained weights. The rest of the parameters in the model, the image weights and the LLM, are initialized from corresponding models following their finetuning stages. Similar to video pre-training, we then finetune only the video parameters on the video SFT data. For this stage, we increase the video length to 64 frames, and use an aggregation factor of 32 to get two effective frames. The resolution of the chunks is also increased to be consistent with the corresponding image hyperparameters.
7.5.3 Preference Data
We built multimodal pair-wise preference datasets for reward modeling and direct preference optimization.
• Human annotations. The human-annotated preference data consists of comparisons between two different model outputs, labeled as “chosen” and “rejected”, with 7-scale ratings. The models used to generate responses are sampled on-the-fly from a pool of the best recent models, each with different characteristics. We update the model pool weekly. Besides preference labels, we also request annotators to provide optional human edits to correct inaccuracies in “chosen” responses because vision tasks have a low tolerance for inaccuracies. Note that human editing is an optional step because there is a trade-off between volume and quality in practice.
• Synthetic data. Synthetic preference pairs could also be generated by using text-only LLMs to edit and deliberately introduce errors in the supervised finetuning dataset. We took the conversational data as input, and use an LLM to introduce subtle but meaningful errors (e.g., change objects, change attributes, add mistakes in calculations, etc.). These edited responses are used as negative “rejected” samples and paired with the “chosen” original supervised finetuning data.
59


7.5.6 Rejection Sampling

Most available question-answer pairs only contain the final answer and lack the chain-of-thought explanation that is required to train a model that generalizes well for reasoning tasks. We use rejection sampling to generate the missing explanations for such examples and boost the model’s reasoning capabilities.

Given a question-answer pair, we generate multiple answers by sampling the finetuned model with different system prompts or temperature. Next, we compare the generated answers to the ground-truth via heuristics or an LLM judge. Finally, we retrain the model by adding the correct answers back into the finetuning data mix. We find it useful to keep multiple correct answers per question.

To ensure we only add high-quality examples back into training, we implemented the following two guardrails. First, we find that some examples contain incorrect explanations, despite the final answer being correct. We observed that this pattern occurs more frequently for questions where only a small fraction of the generated answers is correct. Therefore, we drop answers for questions where the probability of the answer being correct is below a certain threshold. Second, raters prefer some answers over others due to differences in language or style. We use the reward model to select top-K highest-quality answers and add them back into training.

7.5.7 Quality Tuning

We curate a small but highly selective SFT dataset where all samples have been rewritten and verified either by humans or our best models to meet our highest standards. We train DPO models with this data to improve response quality, calling the process Quality-Tuning (QT). We find that QT significantly improves human evaluations without affecting generalization verified by benchmarks when the QT dataset covers a wide range

60

<table style="width:100%;">
<tr>
<th>Llama 3-V 8B</th>
<th>Llama 3-V 70B</th>
<th>Llama 3-V 405B</th>
<th>GPT-4V</th>
<th>GPT-4o</th>
<th>Gemini 1.5 Pro</th>
<th>Claude 3.5</th>
</tr>
<tr>
<td>49.6</td>
<td>60.6</td>
<td>64.5</td>
<td>56.4</td>
<td>69.1</td>
<td>62.2</td>
<td>68.3</td>
</tr>
<tr>
<td>78.0</td>
<td>79.1</td>
<td>80.2</td>
<td>77.2</td>
<td>-</td>
<td>80.2</td>
<td>-</td>
</tr>
<tr>
<td>84.4</td>
<td>93.0</td>
<td>94.1</td>
<td>78.2</td>
<td>94.2</td>
<td>94.4</td>
<td>94.7</td>
</tr>
<tr>
<td>78.7</td>
<td>83.2</td>
<td>85.8</td>
<td>78.4</td>
<td>85.7</td>
<td>87.2</td>
<td>90.8</td>
</tr>
<tr>
<td>78.2</td>
<td>83.4</td>
<td>84.8</td>
<td>78.0</td>
<td>-</td>
<td>78.7</td>
<td>-</td>
</tr>
<tr>
<td>84.4</td>
<td>92.2</td>
<td>92.6</td>
<td>88.4</td>
<td>92.8</td>
<td>93.1△</td>
<td>95.2</td>
</tr>
</table>
<p>Table 29 Image understanding performance of our vision module attached to Llama 3. We compare model performance to GPT-4V, GPT-4o, Gemini 1.5 Pro, and Claude 3.5 Sonnet.△Results obtained using external OCR tools.</p>
<p>of tasks and proper early stopping is applied. We select checkpoints at this stage purely based on benchmarks to ensure capabilities are retained or improved.</p>
<p>7.6 Image Recognition Results</p>
<p>We evaluate the performance of the image understanding capabilities of Llama 3 on a range of tasks spanning natural image understanding, text understanding, charts understanding and multimodal reasoning:</p>
<ul>
<li>MMMU (Yue et al., 2024a) is a challenging dataset for mulitmodal reasoning where model is expected to understand images and solve college-level problems spanning 30 different disciplines. This includes both multiple-choice and open ended questions. We evaluate our model on the validation set with 900 images, in line with other works.</li>
<li>VQAv2 (Antol et al., 2015) tests the ability of a model to combine image understanding, language understanding and commonsense knowlege to answer generic questions about natural images</li>
<li>AI2 Diagram (Kembhavi et al., 2016) evaluates models capability to parse scientific diagrams and answer questions about the same. We use the same evaluation protocol as Gemini and x.ai, and report scores using a transparent bounding box.</li>
<li>ChartQA (Masry et al., 2022) is a challenging benchmark for charts understanding. This requires model to visually understand different kinds of charts and answer logical questions about the charts.</li>
<li>TextVQA (Singh et al., 2019) is a popular benchmark dataset that requires models to read and reason about text in images to answer questions about them. This tests the OCR understanding ability of the model on natural images.</li>
<li>DocVQA (Mathew et al., 2020) is a benchmark dataset focused on document analysis and recognition. It contains images of a wide range of documents which evaluates a model’s ability to perform OCR understanding and reason about the contents of a document to answer questions about them.</li>
</ul>
<p>Table 29 presents the results of our experiments. The results in the table show that our vision module attached to Llama 3 performs competitively across a wide range of image-recognition benchmarks at varying model capacities. Using the resulting Llama 3-V 405B model, we outperform GPT-4V on all benchmarks, while being slightly behind Gemini 1.5 Pro and Claude 3.5 Sonnet. Llama 3 405B appears particularly competitive on document understanding tasks.</p>
<p>7.7 Video Recognition Results</p>
<p>We evaluate our video adapter for Llama 3 on three benchmarks:</p>
<ul>
<li>Perception Test (Pătrăucean et al., 2023) evaluates the model’s ability to answer temporal reasoning questions focusing on skills (memory, abstraction, physics, semantics) and different types of reasoning (descriptive, explanatory, predictive, counterfactual). It consists of 11.6K test QA pairs, each with an on-average 23s long video, filmed by 100 participants worldwide to show perceptually interesting tasks. We focus on the multiple-choice question answering task, where each question is paired with</li>
</ul>
<p>61</p>
<html>
<table border="1">
<tr>
<th></th>
<th>Llama 3-V 8B</th>
<th>Llama 3-V 70B</th>
<th>Gemini 1.0 Pro</th>
<th>Gemini 1.0 Ultra</th>
<th>Gemini 1.5 Pro</th>
<th>GPT-4V</th>
<th>GPT-4o</th>
</tr>
<tr>
<td>Perception Test (test)</td>
<td>53.8</td>
<td>60.8</td>
<td>51.1</td>
<td>54.7</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>TVQA (val)</td>
<td>82.5</td>
<td>87.9</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>87.3</td>
<td>-</td>
</tr>
<tr>
<td>NExT-QA (test)</td>
<td>27.3</td>
<td>30.3</td>
<td>28.0</td>
<td>29.9</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>ActivityNet-QA (test)</td>
<td>52.7</td>
<td>56.3</td>
<td>49.8</td>
<td>52.2</td>
<td>57.5</td>
<td>-</td>
<td>61.9</td>
</tr>
</table>
<p>Table 30 Video understanding performance of our vision module attached to Llama 3. We find that across range of tasks covering long-form and temporal video understanding, our vision adapters for Llama3 8B and 70B parameters are competitive and sometimes even outperform alternative models.</p>
<p>three possible options. We report performance on the held-out test split which is accessed by submitting our predictions to an online challenge server.<sup>16</sup></p>
<p>• NExT-QA (Xiao et al., 2021) is another temporal and causal reasoning benchmark, with a focus on open-ended question answering. It consists of 1K test videos each on-average 44s in length, paired with 9K questions. The evaluation is performed by comparing the model's responses with the ground truth answer using Wu-Palmer Similarity (WUPS) (Wu and Palmer, 1994).<sup>17</sup></p>
<p>• TVQA (Lei et al., 2018) evaluates the model's ability to perform compositional reasoning, requiring spatiotemporal localization of relevant moments, recognition of visual concepts, and joint reasoning with subtitle-based dialogue. This dataset, being derived from popular TV shows, additionally tests for the model's ability to leverage its outside-knowledge of those TV shows in answering the questions. It consists of over 15K validation QA pairs, with each corresponding video clip being on-average 76s in length. It also follows a multiple-choice format with five options for each question, and we report performance on the validation set following prior work (OpenAI, 2023b).</p>
<p>• ActivityNet-QA (Yu et al., 2019) evaluates the model's ability to reason over long video clips to understand actions, spatial relations, temporal relations, counting, etc. It consists of 8K test QA pairs from 800 videos, each on-average 3 minutes long. For evaluation, we follow the protocol from prior work (Google, 2023; Lin et al., 2023; Maaz et al., 2024), where the model generates short one-word or one-phrase answers, and the correctness of the output is evaluated using the GPT-3.5 API which compares it to the ground truth answer. We report the average accuracy as evaluated by the API.</p>
<p>When performing inference, we uniformly sample frames from the full video clip and pass those frames into the model with a short text prompt. Since most of our benchmarks involve answering multiple-choice questions, we use the following prompt: Select the correct answer from the following options: {question}. Answer with the correct option letter and nothing else. For benchmarks that require producing a short answer (e.g., ActivityNet-QA and NExT-QA), we use the following prompt: Answer the question using a single word or phrase. {question}. For NExT-QA, since the evaluation metric (WUPS) is sensitive to the length and the specific words used, we additionally prompt the model to be specific and respond with the most salient answer, for instance specifying “living room” instead of simply responding with “house” when asked a location question. For benchmarks that contain subtitles (i.e., TVQA), we include the subtitles corresponding to the clip in the prompt during inference.</p>
<p>We present the performance of Llama 3 8B and 70B in Table 30. We compare Llama 3's performance with that of two Gemini and two GPT-4 models. Note that all our results are zero-shot, as we do not include any part of these benchmarks in our training or finetuning data. We find that our Llama 3 models that train a small video adapter during post-training are very competitive, and in some cases even better, than other models that potentially leverage native multimodal processing all the way from pre-training. Llama 3 performs particularly well on video recognition given that we only evaluate the 8B and 70B parameter models. Llama 3 achieves its best performance on PerceptionTest, suggesting the model has a strong ability to perform complex temporal reasoning. On long-form activity understanding tasks like ActivityNet-QA, Llama 3 is able to obtain strong results even though it is processing only up to 64 frames, which means that for a 3-minute long video the model only processes one frame every 3 seconds.</p>
<p><sup>16</sup>See <a href="https://eval.ai/web/challenges/challenge-page/2091/overview">https://eval.ai/web/challenges/challenge-page/2091/overview</a>.</p>
<p><sup>17</sup>See <a href="https://github.com/doc-doc/NExT-OE">https://github.com/doc-doc/NExT-OE</a>.</p>
<p>62</p>

</html>
Figure 29 Architecture of our speech interface for Llama 3.

8 Speech Experiments

We perform experiments to study a compositional approach of integrating speech capabilities into Llama 3, resembling the method we used for visual recognition. On the input side, an encoder, together with an adapter, is incorporated to process speech signals. We leverage a system prompt (in text) to enable different modes of operation for speech understanding in Llama 3. If no system prompt is provided, the model acts as a general-purpose spoken dialogue model which can effectively respond to the user speech in a manner that is consistent with the text-only version of Llama 3. The dialogue history is introduced as the prompt prefix to improve the multi-round dialogue experience. We also experiment with system prompts that enable the use of Llama 3 for automatic speech recognition (ASR) and automatic speech translation (AST). The speech interface of Llama 3 supports up to 34 languages.<sup>18</sup> It also allows for the interleaved input of text and speech, enabling the model to solve advanced audio-comprehension tasks.

We also experiment with a speech generation approach in which we implement a streaming text-to-speech (TTS) system that generates speech waveforms on-the-fly during language model decoding. We design the speech generator for Llama 3 based on a proprietary TTS system and do not fine-tune the language model for speech generation. Instead, we focus on improving speech synthesis latency, accuracy, and naturalness by leveraging Llama 3 embeddings at inference time. The speech interface is illustrated in Figure 28 and 29.

8.1 Data

8.1.1 Speech Understanding

The training data can be categorized into two types. The pre-training data includes a large amount of unlabeled speech, which is used to initialize the speech encoder in a self-supervised manner. The supervised finetuning data includes speech recognition, speech translation, and spoken dialogue data; this data is used to unlock specific abilities when integrated with the large language model.

Pre-training data. To pre-train the speech encoder, we curate a dataset of approximately 15M hours of speech recordings encompassing a large number of languages. We filter our audio data using a voice activity detection (VAD) model and select audio samples with a VAD threshold above 0.7 for pre-training. In speech pre-training data, we also focus on ensuring the absence of PII. We use the Presidio Analyzer to identify such PII.

Speech recognition and translation data. Our ASR training data contains 230K hours of manually transcribed speech recordings that span 34 languages. Our AST training data contains 90K hours of translations in two directions: from 33 languages to English and from English to 33 languages. This data contains both supervised and synthetic data generated using the NLLB toolkit (NLLB Team et al., 2022). The use of synthetic AST data enables us to increase model quality for low-resource languages. The speech segments in our data have a maximum length of 60 seconds.

Spoken dialogue data. To finetune the speech adapter for spoken dialogue, we synthetically generate responses

<sup>18</sup>The speech interface supports the following 34 languages: Arabic, Bengali, Chinese, Czech, Dutch, English, Finnish, French, German, Greek, Gujarati, Hindi, Hungarian, Indonesian, Italian, Japanese, Kannada, Korean, Malayalam, Marathi, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Swahili, Swedish, Tamil, Telugu, Thai, Turkish, Urdu, Vietnamese.

63

8.2 Model Architecture

8.2.1 Speech Understanding

On the input side, the speech module consists of two successive modules: a speech encoder and an adapter. The output of the speech module is directly fed into the language model as token representation, enabling direct interaction between speech and text tokens. Furthermore, we incorporate two new special tokens to enclose the sequence of speech representations. The speech module differs substantially from the vision module (see Section 7), which feeds multi-modal information into the language model via cross-attention layers. By contrast, the speech module generates embeddings that can be seamlessly integrated with text tokens, enabling the speech interface to leverage all the capabilities of the Llama 3 language model.

Speech encoder. Our speech encoder is a Conformer (Gulati et al., 2020) model with 1B parameters. The input to the model consists of 80-dimensional mel-spectrogram features, which are first processed by a stride-4 stacking layer followed by a linear projection to reduce the frame length to 40 ms. The resulting features are processed by an encoder with 24 Conformer layers. Each Conformer layer has a latent dimension of 1536, and consists of two Macron-net style feed-forward networks with dimension 4096, a convolution module with kernel size 7, and a rotary attention module (Su et al., 2024) with 24 attention heads.

Speech adapter. The speech adapter contains about 100M parameters. It is composed of a convolution layer, a rotary Transformer layer, and a linear layer. The convolution layer has a kernel size of 3 and a stride of 2, which is designed to reduce the speech frame length to 80 ms. This allows the model to provide more coarse-grained features to the language model. The Transformer layer has a latent dimension of 3072 and a feed-forward network with a dimension of 4096 which further processes the information from speech with context after the convolutional downsampling. Finally, the linear layer maps the output dimension to match that of the language-model embedding layer.

64

8.2.2 Speech Generation

We use Llama 3 8B embeddings in two key components for speech generation: Text Normalization and Prosody Modeling. The TN module ensures semantic correctness by contextually transforming written text into spoken form. The PM module enhances naturalness and expressiveness by predicting prosodic features using these embeddings. Together, they enable accurate and natural speech generation.

Text normalization. As a determinant of the semantic correctness of generated speech, the text normalization (TN) module carries out context-aware transformation from written-form text into the respective spoken form which is eventually verbalized by the downstream components. For example, the written-form text 123 is read as a cardinal number (one hundred twenty three) or spelled digit-by-digit (one two three) depending on the semantic context. The TN system consists of a streaming LSTM-based sequence-tagging model that predicts the sequence of handcrafted TN rules used to transform the input text (Kang et al., 2024). The neural model also takes in Llama 3 embeddings via cross attention to leverage the contextual information encoded therein, enabling minimal text token lookahead and streaming input/output.

Prosody modeling. To enhance the naturalness and expressiveness of synthesized speech, we integrate a decoder-only Transformer-based Prosody model (PM) (Radford et al., 2021) that takes the Llama 3 embeddings as an additional input. This integration leverages the linguistic capabilities of Llama 3, utilizing both its textual output and intermediate embeddings at the token rate (Devlin et al., 2018; Dong et al., 2019; Raffel et al., 2020; Guo et al., 2023) to enhance the prediction of prosody features, thus reducing the lookahead required by the model.

The PM integrates several input components to generate comprehensive prosody predictions: linguistic features derived from the text normalization front-end detailed above, tokens, and embeddings. The PM predicts three key prosodic features: log duration of each phone, log F0 (fundamental frequency) average, and log power average across the phone duration. The model comprises a uni-directional Transformer and six attention heads. Each block includes cross-attention layers and dual fully connected layers with a hidden dimension of 864. A distinctive feature of the PM is its dual cross-attention mechanism, with one layer dedicated to linguistic inputs and the other to Llama embeddings. This setup efficiently manages varying input rates without requiring explicit alignment.

8.3 Training Recipe

8.3.1 Speech Understanding

Training of the speech module is done in two stages. The first stage, speech pre-training, leverages unlabeled data to train a speech encoder that exhibits strong generalization capabilities across languages and acoustic conditions. In the second stage, supervised fine-tuning, the adapter and pre-trained encoder are integrated with the language model, and trained jointly with it while the LLM stays frozen. This enables the model to respond to speech input. This stage uses labeled data corresponding to speech understanding abilities.

Multilingual ASR and AST modeling often results in language confusion/interference, which leads to degraded performance. A popular way to mitigate this is to incorporate language identification (LID) information, both on the source and target side. This can lead to improved performance in the predetermined set of directions, but it does come with potential loss of generality. For instance, if a translation system expects LID on both source and target side, then the model will not likely to show good zero-shot performance in directions that were not seen in training. So our challenge is to design a system that allows LID information to some extent, but keeps the model general enough such that we can have the model do speech translation in unseen directions. To address this, we design system prompts which only contain LID for the text to be emitted (target side). There is no LID information for the speech input (source side) in these prompts, which also potentially allows it to work with code-switched speech. For ASR, we use the following system prompt: Repeat after me in {language}:, where {language} comes from one of the 34 languages (English, French, etc.) For speech translation, the system prompt is: Translate the following sentence into {language}:. This design has been shown to be effective in prompting the language model to respond in the desired language. We used the same system prompts during training and inference.

Speech pre-training. We use the self-supervised BEST-RQ algorithm (Chiu et al., 2022) to pre-train the speech
65

8.4 Speech Understanding Results

We evaluate the speech understanding capabilities of our speech interface for Llama 3 on three tasks: (1) automatic speech recognition, (2) speech translation, and (3) spoken question answering. We compare the performance of our speech interface for Llama 3 with three state-of-the-art models for speech understanding: Whisper (Radford et al., 2023), SeamlessM4T (Barrault et al., 2023), and Gemini.<sup>19</sup> In all the evaluations, we used greedy search for Llama 3 token prediction.

Speech recognition. We evaluate the ASR performance on the English datasets of Multilingual LibriSpeech (MLS; Pratap et al. (2020)), LibriSpeech (Panayotov et al., 2015), VoxPopuli (Wang et al., 2021a), and a subset of the multilingual FLEURS dataset (Conneau et al., 2023). In evaluation, the decoding results are post-processed using the Whisper text normalizer to ensure consistency in comparing with the reported results of other models. On all benchmarks, we measure the word error rate of our speech interface for Llama 3

<sup>19</sup>Due to technical limitations, we compare with the performance of Gemini on MLS reported in the original paper.

66

<table style="width:100%;">
<tr>
<th></th>
<th>Llama 3 8B</th>
<th>Llama 3 70B</th>
<th>Whisper</th>
<th>SeamlessM4T v2</th>
<th>Gemini 1.0 Ultra</th>
<th>Gemini 1.5 Pro</th>
</tr>
<tr>
<td>MLS (English)</td>
<td>4.9</td>
<td>4.4</td>
<td>6.2 (v2)</td>
<td>6.5</td>
<td>4.4</td>
<td>4.2</td>
</tr>
<tr>
<td>LibriSpeech (test-other)</td>
<td>3.4</td>
<td>3.1</td>
<td>4.9 (v2)</td>
<td>6.2</td>
<td>–</td>
<td>–</td>
</tr>
<tr>
<td>VoxPopuli (English)</td>
<td>6.2</td>
<td>5.7</td>
<td>7.0 (v2)</td>
<td>7.0</td>
<td>–</td>
<td>–</td>
</tr>
<tr>
<td>FLEURS (34 languages)</td>
<td>9.6</td>
<td>8.2</td>
<td>14.4 (v3)</td>
<td>11.7</td>
<td>–</td>
<td>–</td>
</tr>
</table>
<p>Table 31 Word error rate of our speech interface for Llama 3 on speech recognition tasks. We report the performance of Whisper, SeamlessM4T, and Gemini for reference.</p>
<table style="width:100%;">
<tr>
<th></th>
<th>Llama 3 8B</th>
<th>Llama 3 70B</th>
<th>Whisper v2</th>
<th>SeamlessM4T v2</th>
</tr>
<tr>
<td>FLEURS (33 lang. → English)</td>
<td>29.5</td>
<td>33.7</td>
<td>21.9</td>
<td>28.6</td>
</tr>
<tr>
<td>Covost 2 (15 lang. → English)</td>
<td>34.4</td>
<td>38.8</td>
<td>33.8</td>
<td>37.9</td>
</tr>
</table>
<p>Table 32 BLEU score of our speech interface for Llama 3 on speech translation tasks. We report the performance of Whisper and SeamlessM4T for reference.</p>
<p>on the standard test set of those benchmarks, except for Chinese, Japanese, Korean and Thai, where the character error rate is reported.</p>
<p>Table 31 shows the results of ASR evaluations. It demonstrates the strong performance of Llama 3 (and multi-modal foundation models more generally) on speech recognition tasks: our model outperforms models that are tailored to speech like Whisper<sup>20</sup> and SeamlessM4T on all benchmarks. On MLS English, Llama 3 performs similarly to Gemini.</p>
<p>Speech translation. We also evaluate our models on speech translation tasks in which the model is asked to translate non-English speech into English text. We use the FLEURS and Covost 2 (Wang et al., 2021b) datasets in these evaluations, measuring BLEU scores of the translated English. Table 32 presents the results of these experiments.<sup>21</sup> The performance of our models in speech translation highlights the advantages of multimodal foundation models for tasks such as speech translation.</p>
<p>Spoken question answering. The speech interface of Llama 3 demonstrates remarkable question answering capabilities. The model can effortlessly comprehend code-switched speech without any prior exposure to such data. Notably, although the model was trained only on single-turn dialogue, it is capable of engaging in extended, coherent multi-turn dialogue sessions. Figure 30 presents a few examples that highlight these multilingual and multi-turn capabilities.</p>
<p>Safety. We evaluate the safety of our speech model on MuTox (Costa-jussà et al., 2023), a multilingual audio-based dataset of 20,000 utterances for English and Spanish and 4,000 for 19 other languages, each with toxicity labels attached. The audio is passed as input to the model and the output is evaluated for toxicity, after cleaning some special characters. We apply the MuTox classifier (Costa-jussà et al., 2023) and compare the results with Gemini 1.5 Pro. We evaluate the percentage of added toxicity (AT), when the input prompt is safe and the output is toxic, and the percentage of lost toxicity (LT), when the input prompt is toxic and the answer is safe. Table 33 shows the results for English and an average across all 21 languages that we evaluated on.<sup>22</sup> The percentage of added toxicity is very low: our speech models have the lowest percentage of added toxicity for English, with less than 1%. It removes significantly more toxicity than it adds.</p>
<p>8.5 Speech Generation Results</p>
<p>For speech generation, we focus on evaluating the quality of token-wise input streaming models with the Llama 3 embeddings for the text normalization and prosody modeling tasks. The evaluation focuses on</p>
<p><sup>20</sup>On FLEURS ASR, Malayalam is not officially reported for Whisper v3, so we use the average of 33 languages.</p>
<p><sup>21</sup>On Covost 2, we evaluate only on 15 (out of 21) languages.</p>
<p><sup>22</sup>Note that for Gemini, we encountered that a significant number of responses were empty, which could be due to safety filters on their side (though some empty responses were for non-toxic input) or to rate limits. To conduct the analysis, we assumed that all the empty responses are safe. This is the most conservative approach for results and the upper bound of what Gemini results would look like.</p>
<p>67</p>


Figure 30 Transcribed dialogue examples using the speech interface for Llama 3. The examples illustrate zero-shot multi-turn and code-switching capabilities.

<br>

<table>
  <tr>
    <th>Language</th>
    <th>Llama 3 8B</th>
    <th>Llama 3 70B</th>
    <th>Gemini 1.5 Pro</th>
  </tr>
  <tr>
    <td>AT (↓)</td>
    <td>LT (↑)</td>
    <td>AT (↓)</td>
    <td>LT (↑)</td>
    <td>AT (↓)</td>
    <td>LT (↑)</td>
  </tr>
  <tr>
    <td>English</td>
    <td>0.84</td>
    <td>15.09</td>
    <td>0.68</td>
    <td>15.46</td>
    <td>1.44</td>
    <td>13.42</td>
  </tr>
  <tr>
    <td>Overall</td>
    <td>2.31</td>
    <td>9.89</td>
    <td>2.00</td>
    <td>10.29</td>
    <td>2.06</td>
    <td>10.94</td>
  </tr>
</table>

Table 33 Speech toxicity of our speech interface to Llama 3 on the MuTox dataset. AT refers to added toxicity (%) and LT refers to lost toxicity (%).

<br>

comparisons with models that do not take the Llama 3 embeddings as an additional input.

<br>

Text normalization. To measure the effect of Llama 3 embeddings, we experimented with changing the amount of right context the model uses. We trained the model using a right context of 3 TN tokens (demarcated by unicode category). This model is compared to models that do not use the Llama 3 embeddings, using a 3-token right context or a full bi-directional context. As expected, Table 34 shows using the full right context improves performance for the model without Llama 3 embeddings. However, the model that incorporates the Llama 3 embeddings outperforms all other models, hence enabling token-rate input/output streaming without relying on long context in the input.

<br>

<table>
  <tr>
    <th>Model</th>
    <th>Context</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td>Without Llama 3 8B</td>
    <td>3</td>
    <td>73.6%</td>
  </tr>
  <tr>
    <td>Without Llama 3 8B</td>
    <td>∞</td>
    <td>88.0%</td>
  </tr>
  <tr>
    <td>With Llama 3 8B</td>
    <td>3</td>
    <td>90.7%</td>
  </tr>
</table>

Table 34 Sample-wise text normalization (TN) accuracy. We compare models with or without Llama 3 8B embeddings, and using different right-context values.

<br>

Prosody modeling. To evaluate the performance of the our prosody model (PM) with Llama 3 8B, we conducted two sets of human evaluation comparing models with and without Llama 3 embeddings. Raters listened to samples from different models and indicated their preferences. To generate the final speech waveform, we use an in-house transformer based acoustic model (Wu et al., 2021) that predicts spectral features and a WaveRNN neural vocoder (Kalchbrenner et al., 2018) to generate the final speech waveform.

<br>

First, we compare directly to a streaming baseline model without Llama 3 embeddings. In the second test, the Llama 3 8B PM is compared to a non-streaming baseline model without Llama 3 embeddings. As shown in Table 35, the Llama 3 8B PM is preferred 60% of the time compared to the streaming baseline, and

<br>

68

<html><table><tr><td>Model</td><td>Preference</td></tr><tr><td>PM for Llama 3 8B</td><td>60.0%</td></tr><tr><td>Streaming phone-only baseline</td><td>40.0%</td></tr></table><html><table><tr><td>Model</td><td>Preference</td></tr><tr><td>PM for Llama 3 8B</td><td>63.6%</td></tr><tr><td>Non-streaming phone-only baseline</td><td>36.4%</td></tr></table>Table 35 Prosody Modeling (PM) evaluation. Left: Rater preferences of PM for Llama 3 8B vs. streaming phone-only baseline. Right: Rater preferences of PM for Llama 3 8B vs. non-streaming phone-only baseline.

63.6% of the time compared to the non-streaming baseline, indicating a significant improvement in perceived quality. The key advantage of the Llama 3 8B PM is its token-wise streaming capability (Section 8.2.2), which maintains low latency during inference. This reduces the model’s lookahead requirements, enabling more responsive and real-time speech synthesis compared to non-streaming baselines. Overall, the Llama 3 8B prosody model consistently outperforms the baseline models, demonstrating its effectiveness in enhancing the naturalness and expressiveness of synthesized speech.

9 Related Work

The development of Llama 3 builds on a large body of prior work studying foundation models for language, images, videos, and speech. A comprehensive overview of that work is outside the scope of this paper; we refer the reader to Bordes et al. (2024); Madan et al. (2024); Zhao et al. (2023a) for such overviews. Below, we briefly outline seminal works that directly influenced the development of Llama 3.

9.1 Language

Scale. Llama 3 follows the enduring trend of applying straightforward methods at ever increasing scales in foundation models. Improvements are driven by increased compute and improved data, with the 405B model using almost fifty times the pre-training compute budget of Llama 2 70B. Despite containing 405B parameters, our largest Llama 3 in fact contains fewer parameters than earlier and much less performant models such as PALM (Chowdhery et al., 2023), due to better understanding of scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022). Little is publicly known about the size of other frontier models, such as Claude 3 or GPT 4 (OpenAI, 2023a), but overall performance is comparable.

Small models. Developments in smaller models have paralleled those in large models. Models with fewer parameters can dramatically improve inference cost and simplify deployment (Mehta et al., 2024; Team et al., 2024). The smaller Llama 3 models achieve this by training far beyond the point of compute optimal training, effectively trading training compute for inference efficiency. An alternative path is to distill larger models into smaller ones, as in Phi (Abdin et al., 2024).

Architectures. While Llama 3 makes minimal architectural modifications compared to Llama 2, other recent foundation models have explored other designs. Most notably, mixture of experts architectures (Shazeer et al., 2017; Lewis et al., 2021; Fedus et al., 2022; Zhou et al., 2022) can be used as an efficient way to increase the capacity of a model, such as in Mixtral (Jiang et al., 2024) and Arctic (Snowflake, 2024). Llama 3 outperforms these models, suggesting that dense architectures are not the limiting factor, but there remain numerous trade-offs in terms of training and inference efficiency, and model stability at scale.

Open source. Open weights foundation models have rapidly improved over the last year, with Llama3-405B now competitive with the current closed weight state-of-the-art. Numerous model families have recently been developed, including Mistral (Jiang et al., 2023), Falcon (Almazrouei et al., 2023), MPT (Databricks, 2024), Pythia (Biderman et al., 2023), Arctic (Snowflake, 2024), OpenELM (Mehta et al., 2024), OLMo (Groeneveld et al., 2024), StableLM (Bellagente et al., 2024), OpenLLaMA (Geng and Liu, 2023), Qwen (Bai et al., 2023), Gemma (Team et al., 2024), Grok (XAI, 2024), and Phi (Abdin et al., 2024).

Post-training. Post-training Llama 3 follows the established strategy of instruction tuning (Chung et al., 2022; Ouyang et al., 2022) followed by alignment with human feedback (Kaufmann et al., 2023). While some studies have shown the surprising effectiveness of lightweight alignment procedures (Zhou et al., 2024), Llama 3 uses millions of human instructions and preference judgments to improve the pre-trained model, including

69

techniques such as rejection sampling (Bai et al., 2022), supervised finetuning (Sanh et al., 2022), and Direct Preference Optimization (Rafailov et al., 2023). In order to curate these instruction and preference examples, we deploy earlier versions of Llama 3 to filter (Liu et al., 2024c), re-write (Pan et al., 2024), or generate prompts and responses (Liu et al., 2024b) and apply these techniques through multiple rounds of post-training.

9.2 Multimodality

Our experiments with multimodal capabilities for Llama 3 are part of a long line of work on foundation models that jointly model multiple modalities.

Images. A substantial body of work has trained image-recognition models on large amounts of image-text pairs, for example, Mahajan et al. (2018); Xiao et al. (2024a); Team (2024); OpenAI (2023b). Radford et al. (2021) presented one of the first models to jointly embed images and text via contrastive learning. More recently, a series of models has studied approaches similar to the one used in Llama 3, for example, Alayrac et al. (2022); Dai et al. (2023); Liu et al. (2023c,b); Yang et al. (2023b); Ye et al. (2023); Zhu et al. (2023). Our approach in Llama 3 combines ideas from many of these papers to achieve results that are comparable with Gemini 1.0 Ultra (Google, 2023) and GPT-4 Vision (OpenAI, 2023b); see Section 7.6.

Video. Although video inputs are supported by an increasing number of foundation models (Google, 2023; OpenAI, 2023b), the body of work on joint modeling of videos and language is not that large. Akin to Llama 3, most current studies adopt an adapter approach to align video and language representations and unlock question-answering and reasoning about videos (Lin et al., 2023; Li et al., 2023a; Maaz et al., 2024; Zhang et al., 2023; Zhao et al., 2022). We find that such approaches produce results that are competitive with the state-of-the-art; see Section 7.7.

Speech. Our work also fits in a larger body of work combining language and speech modeling. Earlier joint models of text and speech include AudioPaLM (Rubenstein et al., 2023), VioLA (Wang et al., 2023b), VoxtLM Maiti et al. (2023), SUTLM (Chou et al., 2023), and Spirit-LM (Nguyen et al., 2024). Our work builds on prior compositional approaches to combining speech and language like Fathullah et al. (2024). Unlike most prior work, we opt to not finetune the language model itself for speech tasks as doing so may lead to contention on non-speech tasks. We find that at larger model scales, strong performances are attainable even without such finetuning; see Section 8.4.

10 Conclusion

In many ways, the development of high-quality foundation models is still in its infancy. Our experience in developing Llama 3 suggests that substantial further improvements of these models are on the horizon. Throughout the development of the Llama 3 model family, we found that a strong focus on high-quality data, scale, and simplicity consistently yielded the best results. In preliminary experiments, we explored more complex model architectures and training recipes but did not find the benefits of such approaches to outweigh the additional complexity they introduce in model development.

Developing a flagship foundation model such as Llama 3 involves overcoming a plethora of deep technical problems but also requires clever organizational decisions. For example, to ensure Llama 3 is not accidentally overfitted on commonly used benchmarks, our pre-training data was procured and processed by a separate team that was strongly incentivized to prevent contamination of that pre-training data with external benchmarks. As another example, we ensure that our human evaluations remain trustworthy by allowing only a small set of researchers who do not contribute to model development to perform and access these evaluations. While such organizational decisions are rarely discussed in technical papers, we found them to be pivotal to the successful development of the Llama 3 family of models.

We shared the details of our development process because we believe this will: (1) help the larger research community understand the key factors of foundation model development and (2) contribute to a more informed debate about the future of foundation models in the general public. We also shared preliminary experiments with integrating multimodal capabilities into Llama 3. While these models are still under active development and not yet ready for release, we hope sharing our results early will accelerate research in this direction.

70

Following the positive outcomes of the detailed safety analyses presented in this paper, we publicly release our Llama 3 language models in order to accelerate the development of AI systems for a plethora of societally relevant use cases and enable the research community to scrutinize our models and identify ways to make these models better and safer. We believe that the public release of foundation models plays a key role in the responsible development of such models, and we hope that the release of Llama 3 encourages the industry to embrace the open, responsible development of AGI.

71

Contributors and Acknowledgements

Llama 3 is the result of the work of a large number of people at Meta. Below, we list all core contributors (people who worked on Llama 3 for at least 2/3rd of the runtime of the project) and contributors (people who worked on Llama 3 for at least 1/5th of the runtime of the project). We list all contributors in alphabetical order of first name.

Core Contributors

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vítor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, and Zoe Papakipos.

Contributors

Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani

72

Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi (Jack) Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu (Sid) Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma.

Acknowledgements

We thank Mark Zuckerberg, Chris Cox, Ahmad Al-Dahle, Santosh Janardhan, Joelle Pineau, Yann LeCun, Aparna Ramani, Yee Jiun Song, and Ash Jhaveri for their invaluable support for Llama 3.

We also thank Aasish Pappu, Adebissy Tharinger, Adnan Aziz, Aisha Iqbal, Ajit Mathews, Albert Lin, Amar Budhiraja, Amit Nagpal, Andrew Or, Andrew Prasetyo Jo, Ankit Jain, Antonio Prado, Aran Mun, Armand Kok, Ashmitha Jeevaraj Shetty, Aya Ibrahim, Bardiya Sadeghi, Beibei Zhu, Bell Praditchai, Benjamin Muller, Botao Chen, Carmen Wang, Carolina Tsai, Cen Peng, Cen Zhao, Chana Greene, Changsheng Zhao, Chenguang Zhu, Chloé Bakalar, Christian Fuegen, Christophe Ropers, Christopher Luc, Dalton Flanagan, Damien Sereni, Dan Johnson, Daniel Haziza, Daniel Kim, David Kessel, Digant Desai, Divya Shah, Dong Li, Elisabeth Michaels, Elissa Jones, Emad El-Haraty, Emilien Garreau, Eric Alamillo, Eric Hambro, Erika Lal, Eugen Hotaj, Fabian Gloeckle, Fadli Basyari, Faith Eischen, Fei Kou, Ferdi Adeputra, Feryandi Nurdiantoro, Flaurencya Ciputra, Forest Zheng, Francisco Massa, Furn Techaletumpai, Gobinda Saha, Gokul Nadathur,

73

Greg Steinbrecher, Gregory Chanan, Guille Cobo, Guillem Brasó, Hany Morsy, Haonan Sun, Hardik Shah, Henry Erksine Crum, Hongbo Zhang, Hongjiang Lv, Hongye Yang, Hweimi Tsou, Hyunbin Park, Ian Graves, Jack Wu, Jalpa Patel, James Beldock, James Zeng, Jeff Camp, Jesse He, Jilong Wu, Jim Jetsada Machom, Jinho Hwang, Jonas Gehring, Jonas Kohler, Jose Leitao, Josh Fromm, Juan Pino, Julia Rezende, Julian Garces, Kae Hansanti, Kanika Narang, Kartik Khandelwal, Keito Uchiyama, Kevin McAlister, Kimish Patel, Kody Bartelt, Kristina Pereyra, Kunhao Zheng, Lien Thai, Lu Yuan, Lunwen He, Marco Campana, Mariana Velasquez, Marta R. Costa-jussa, Martin Yuan, Max Ren, Mayank Khamesra, Mengjiao MJ Wang, Mengqi Mu, Mergen Nachin, Michael Suo, Mikel Jimenez Fernandez, Mustafa Ozdal, Na Li, Nahiyan Malik, Naoya Miyanohara, Narges Torabi, Nathan Davis, Nico Lopero, Nikhil Naik, Ning Li, Octary Azis, PK Khambanonda, Padchara Bubphasan, Pian Pawakapan, Prabhav Agrawal, Praveen Gollakota, Purin Waranimman, Qian Sun, Quentin Carbonneaux, Rajasi Saha, Rhea Nayak, Ricardo Lopez-Barquilla, Richard Huang, Richard Qiu, Richard Tosi, Rishi Godugu, Rochit Sapra, Rolando Rodriguez Antunez, Ruihan Shan, Sakshi Boolchandani, Sam Corbett-Davies, Samuel Djunaedi, Sarunya Pumma, Saskia Adams, Scott Wolchok, Shankar Kalyanaraman, Shashi Gandham, Shengjie Bi, Shengxing Cindy, Shervin Shahidi, Sho Yaida, Shoubhik Debnath, Sirirut Sonjai, Srikanth Sundaresan, Stephanie Worland, Susana Contrera, Tejas Shah, Terry Lam, Tony Cao, Tony Lee, Tristan Rice, Vishy Poosala, Wenyu Chen, Wesley Lee, William Held, Xiaozhu Meng, Xinhua Wang, Xintian Wu, Yanghan Wang, Yaroslava Kuzmina, Yifan Wang, Yuanhao Xiong, Yue Zhao, Yun Wang, Zaibo Wang, Zechun Liu, and Zixi Qi for helpful contributions to Llama 3.

74

References

Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, and Ari S Morcos. Semdedup: Data-efficient learning at web-scale through semantic deduplication. arXiv preprint arXiv:2303.09540, 2023.

Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219, 2024.

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan. Flamingo: a visual language model for few-shot learning. arXiv preprint arXiv:2204.14198, 2022.

Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Mérouane Debbah, Étienne Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic, et al. The falcon series of open language models. arXiv preprint arXiv:2311.16867, 2023.

Norah Alzahrani, Hisham Abdullah Alyahya, Yazeed Alnumay, Sultan Alrashed, Shaykhah Alsubaie, Yusef Almushaykeh, Faisal Mirza, Nouf Alotaibi, Nora Al-Twairesh, Areeb Alowisheq, M. Saiful Bari, and Haidar Khan. When benchmarks are targets: Revealing the sensitivity of large language model leaderboards. CoRR, abs/2402.01781, 2024. doi: 10.48550/ARXIV.2402.01781. https://doi.org/10.48550/arXiv.2402.01781.

Aida Amini, Saadia Gabriel, Peter Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. arXiv preprint arXiv:1905.13319, 2019.

Chenxin An, Shansan Gong, Ming Zhong, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. L-eval: Instituting standardized evaluation for long context language models. arXiv preprint arXiv:2307.11088, 2023a.

Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, Jian-Guang Lou, and Weizhu Chen. Learning from mistakes makes llm better reasoner. arXiv preprint arXiv:2310.20689, 2023b.

Cem Anil, Esin Durmus, Mrinank Sharma, Joe Benton, Sandipan Kundu, Joshua Batson, Nina Rimsky, Meg Tong, Jesse Mu, Daniel Ford, et al. Many-shot jailbreaking. Anthropic, April, 2024.

Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, et al. Pytorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, pages 929–947, 2024.

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. VQA: Visual Question Answering. In International Conference on Computer Vision (ICCV), 2015.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosiute, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemí Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom

75

Brown, and Jared Kaplan. Constitutional AI: harmlessness from AI feedback. CoRR, abs/2212.08073, 2022. doi: 10.48550/ARXIV.2212.08073. https://doi.org/10.48550/arXiv.2212.08073.

Loïc Barrault, Yu-An Chung, Mariano Coria Meglioli, David Dale, Ning Dong, Mark Duppenthaler, Paul-Ambroise Duquenne, Brian Ellis, Hady Elsahar, Justin Haaheim, John Hoffman, Min-Jae Hwang, Hirofumi Inaguma, Christopher Klaiber, Ilia Kulikov, Pengwei Li, Daniel Licht, Jean Maillard, Ruslan Mavlyutov, Alice Rakotoarison, Kaushik Ram Sadagopan, Abinesh Ramakrishnan, Tuan Tran, Guillaume Wenzek, Yilin Yang, Ethan Ye, Ivan Evtimov, Pierre Fernandez, Cynthia Gao, Prangthip Hansanti, Elahe Kalbassi, Amanda Kallet, Artyom Kozhevnikov, Gabriel Mejia Gonzalez, Robin San Roman, Christophe Touret, Corinne Wong, Carleigh Wood, Bokai Yu, Pierre Andrews, Can Balioglu, Peng-Jen Chen, Marta R Costa-jussà, Maha Elbayad, Hongyu Gong, Francisco Guzmán, Kevin Heffernan, Somya Jain, Justine Kao, Ann Lee, Xutai Ma, Alex Mourachko, Benjamin Peloquin, Juan Pino, Sravya Popuri, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Anna Sun, Paden Tomasello, Changhan Wang, Jeff Wang, Skyler Wang, and Mary Williamson. Seamless: Multilingual expressive and streaming speech translation. arXiv preprint arXiv:2312.05187, 2023.

Robin Battey and Sumit Gupta. Training llama: A storage perspective, 2024. https://atscaleconference.com/videos/training-llama-a-storage-perspective/.

Marco Bellagente, Jonathan Tow, Dakota Mahan, Duy Phung, Maksym Zhuravinskyi, Reshinth Adithyan, James Baicoianu, Ben Brooks, Nathan Cooper, Ashish Datta, et al. Stable lm 2 1.6 b technical report. arXiv preprint arXiv:2402.17834, 2024.

Youssef Benchekroun, Megi Dervishi, Mark Ibrahim, Jean-Baptiste Gaya, Xavier Martinet, Grégoire Mialon, Thomas Scialom, Emmanuel Dupoux, Dieuwke Hupkes, and Pascal Vincent. Worldsense: A synthetic benchmark for grounded reasoning in large language models. CoRR, abs/2311.15930, 2023. doi: 10.48550/ARXIV.2311.15930. https://doi.org/10.48550/arXiv.2311.15930.

Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on Freebase from question-answer pairs. In David Yarowsky, Timothy Baldwin, Anna Korhonen, Karen Livescu, and Steven Bethard, editors, Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1533–1544, Seattle, Washington, USA, October 2013. Association for Computational Linguistics. https://aclanthology.org/D13-1160.

Manish Bhatt, Sahana Chennabasappa, Cyrus Nikolaidis, Shengye Wan, Ivan Evtimov, Dominik Gabi, Daniel Song, Faizan Ahmad, Cornelius Aschermann, Lorenzo Fontana, et al. Purple llama cyberseceval: A secure coding benchmark for language models. arXiv preprint arXiv:2312.04724, 2023.

Manish Bhatt, Sahana Chennabasappa, Yue Li, Cyrus Nikolaidis, Daniel Song, Shengye Wan, Faizan Ahmad, Cornelius Aschermann, Yaohui Chen, Dhaval Kapil, et al. Cyberseceval 2: A wide-ranging cybersecurity evaluation suite for large language models. arXiv preprint arXiv:2404.13161, 2024.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pages 2397–2430. PMLR, 2023.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 7432–7439, 2020.

Yuri Bizzoni, Tom S Juzek, Cristina España-Bonet, Koel Dutta Chowdhury, Josef van Genabith, and Elke Teich. How human is machine translationese? comparing human and machine translations of text and speech. In Marcello Federico, Alex Waibel, Kevin Knight, Satoshi Nakamura, Hermann Ney, Jan Niehues, Sebastian Stüker, Dekai Wu, Joseph Mariani, and Francois Yvon, editors, Proceedings of the 17th International Conference on Spoken Language Translation, pages 280–290, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.iwslt-1.34. https://aclanthology.org/2020.iwslt-1.34.

Cody Blakeney, Mansheej Paul, Brett W. Larsen, Sean Owen, and Jonathan Frankle. Does your data spark joy? performance gains from domain upsampling at the end of training, 2024. https://arxiv.org/abs/2406.03476.

Florian Bordes, Richard Yuanzhe Pang, Anurag Ajay, Alexander C. Li, Adrien Bardes, Suzanne Petryk, Oscar Mañas, Zhiqiu Lin, Anas Mahmoud, Bargav Jayaraman, Mark Ibrahim, Melissa Hall, Yunyang Xiong, Jonathan Lebensold, Candace Ross, Srihari Jayakumar, Chuan Guo, Diane Bouchacourt, Haider Al-Tahan, Karthik Padthe, Vasu Sharma, Hu Xu, Xiaoqing Ellen Tan, Megan Richards, Samuel Lavoie, Pietro Astolfi, Reyhane Askari Hemmat, Jun Chen, Kushal Tirumala, Rim Assouel, Mazda Moayeri, Arjang Talattof, Kamalika Chaudhuri, Zechun Liu, Xilun Chen, Quentin Garrido, Karen Ullrich, Aishwarya Agrawal, Kate Saenko, Asli Celikyilmaz, and Vikas Chandra. An introduction to vision-language modeling. 2024.

76

A.Z. Broder. On the resemblance and containment of documents. In Proceedings. Compression and Complexity of SEQUENCES 1997 (Cat. No.97TB100171), pages 21–29, 1997. doi: 10.1109/SEQUEN.1997.666900.

Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P. Meyer, Yuning Chai, Dennis Park, and Yong Jae Lee. Making large multimodal models understand arbitrary visual prompts. In IEEE Conference on Computer Vision and Pattern Recognition, 2024.

Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramèr, and Chiyuan Zhang. Quantifying memorization across neural language models. arXiv:2202.07646, 2022. https://arxiv.org/abs/2202.07646.

Nicolas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, Florian Tramer, Borja Balle, Daphne Ippolito, and Eric Wallace. Extracting training data from diffusion models. In 32nd USENIX Security Symposium (USENIX Security 23), pages 5253–5270, 2023.

Federico Cassano, John Gouwar, Daniel Nguyen, Sydney Nguyen, Luna Phipps-Costin, Donald Pinckney, Ming-Ho Yee, Yangtian Zi, Carolyn Jane Anderson, Molly Q Feldman, Arjun Guha, Michael Greenberg, and Abhinav Jangda. MultiPL-E: A scalable and polyglot approach to benchmarking neural code generation. IEEE Trans. Software Eng., 49(7):3675–3691, 2023.

Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, and Eric Wong. Jailbreaking black box large language models in twenty queries. arXiv preprint arXiv:2310.08419, 2023.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Nuo Chen, Zinan Zheng, Ning Wu, Ming Gong, Yangqiu Song, Dongmei Zhang, and Jia Li. Breaking language barriers in multilingual mathematical reasoning: Insights and observations, 2023. https://arxiv.org/abs/2310.20246.

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588, 2022.

Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Hao Zhang, Banghua Zhu, Michael Jordan, Joseph E Gonzalez, et al. Chatbot arena: An open platform for evaluating LLMs by human preference. arXiv preprint arXiv:2403.04132, 2024.

Chung-Cheng Chiu, James Qin, Yu Zhang, Jiahui Yu, and Yonghui Wu. Self-supervised learning with random-projection quantizer for speech recognition. In International Conference on Machine Learning, pages 3915–3924. PMLR, 2022.

Eunsol Choi, He He, Mohit Iyyer, Mark Yatskar, Wen-tau Yih, Yejin Choi, Percy Liang, and Luke Zettlemoyer. QuAC: Question answering in context. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii, editors, Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2174–2184, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1241. https://aclanthology.org/D18-1241.

Ju-Chieh Chou, Chung-Ming Chien, Wei-Ning Hsu, Karen Livescu, Arun Babu, Alexis Conneau, Alexei Baevski, and Michael Auli. Toward joint language modeling for speech units and text. 2023.

Arnab Choudhury, Yang Wang, Tuomas Pelkonen, Kutta Srinivasan, Abha Jain, Shenghao Lin, Delia David, Siavash Soleimanifard, Michael Chen, Abhishek Yadav, Ritesh Tijoriwala, Denis Samoylov, and Chunqiang Tang. MAST: Global scheduling of ML training across geo-distributed datacenters at hyperscale. In Proceedings from 18th USENIX Symposium on Operating Systems Design and Implementation, 2024.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Y. Zhao, Yanping Huang, Andrew M. Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instruction-finetuned language models. CoRR, abs/2210.11416, 2022. doi: 10.48550/ARXIV.2210.11416. https://doi.org/10.48550/arXiv.2210.11416.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

77

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna. Fleurs: Few-shot learning evaluation of universal representations of speech. In 2022 IEEE Spoken Language Technology Workshop (SLT), pages 798–805, 2023. doi: 10.1109/SLT54892.2023.10023141.

Marta R. Costa-jussà, Mariano Coria Meglioli, Pierre Andrews, David Dale, Prangthip Hansanti, Elahe Kalbassi, Alex Mourachko, Christophe Ropers, and Carleigh Wood. Mutox: Universal multilingual audio-based toxicity dataset and zero-shot detector. 2023.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. 2023.

Databricks. Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs blog. https://www.databricks.com/blog/mpt-7b, 2024.

DeepSeek-AI, Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu, Y. Wu, Yukun Li, Huazuo Gao, Shirong Ma, Wangding Zeng, Xiao Bi, Zihui Gu, Hanwei Xu, Damai Dai, Kai Dong, Liyue Zhang, Yishi Piao, Zhibin Gou, Zhenda Xie, Zhewen Hao, Bingxuan Wang, Junxiao Song, Deli Chen, Xin Xie, Kang Guan, Yuxiang You, Aixin Liu, Qiushi Du, Wenjun Gao, Xuan Lu, Qinyu Chen, Yaohui Wang, Chengqi Deng, Jiashi Li, Chenggang Zhao, Chong Ruan, Fuli Luo, and Wenfeng Liang. Deepseek-coder-v2: Breaking the barrier of closed-source models in code intelligence, 2024. https://arxiv.org/abs/2406.11931.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Aniket Didolkar, Anirudh Goyal, Nan Rosemary Ke, Siyuan Guo, Michal Valko, Timothy Lillicrap, Danilo Rezende, Yoshua Bengio, Michael Mozer, and Sanjeev Arora. Metacognitive capabilities of LLMs: An exploration in mathematical problem solving. arXiv preprint arXiv:2405.12205, 2024.

Li Dong, Nan Yang, Wenhui Wang, Furu Wei, Xiaodong Liu, Yu Wang, Jianfeng Gao, Ming Zhou, and Hsiao-Wuen Hon. Unified language model pre-training for natural language understanding and generation. Advances in neural information processing systems, 32, 2019.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv:2010.11929, 2020.

Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2368–2378, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1246. https://aclanthology.org/N19-1246.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024.

Hany Farid. An overview of perceptual hashing. Journal of Online Trust and Safety, 1(1), 2021.

Yassir Fathullah, Chunyang Wu, Egor Lakomkin, Ke Li, Junteng Jia, Yuan Shangguan, Jay Mahadeokar, Ozlem Kalinli, Christian Fuegen, and Mike Seltzer. Audiochatllama: Towards general-purpose speech abilities for LLMs. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5522–5532, 2024.

William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.

Adithya Gangidi, Rui Miao, Shengbao Zheng, Sai Jayesh Bondu, Guilherme Goes, Hany Morsy, Rohit Puri, Mohammad Riftadi, Ashmitha Jeevaraj Shetty, Jingyi Yang, Shuqiang Zhang, Mikel Jimenez Fernandez, Shashidhar Gandham, and Hongyi Zeng. RDMA over Ethernet for Distributed AI Training at Meta Scale. In ACM Special Interest Group on Data Communication (SIGCOMM), 2024. https://doi.org/10.1145/3651890.3672233.

78

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In International Conference on Machine Learning, pages 10764–10799. PMLR, 2023.

Zorik Gekhman, Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Roi Reichart, and Jonathan Herzig. Does fine-tuning LLMs on new knowledge encourage hallucinations?, 2024.

Xinyang Geng and Hao Liu. Openllama: An open reproduction of llama, 2023. https://github.com/openlm-research/open_llama

Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, and Ishan Misra. Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709, 2023.

Gemini Team Google. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yujiu Yang, Minlie Huang, Nan Duan, Weizhu Chen, et al. Tora: A tool-integrated reasoning agent for mathematical problem solving. arXiv preprint arXiv:2309.17452, 2023.

Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Raghavi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, Will Smith, Emma Strubell, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Luke Zettlemoyer, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah A. Smith, and Hannaneh Hajishirzi. Olmo: Accelerating the science of language models, 2024. https://arxiv.org/abs/2402.00838

Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, et al. Conformer: Convolution-augmented transformer for speech recognition. arXiv preprint arXiv:2005.08100, 2020.

Zhifang Guo, Yichong Leng, Yihan Wu, Sheng Zhao, and Xu Tan. Prompttts: Controllable text-to-speech with text descriptions. In ICASSP 2023–2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.

Vipul Gupta, David Pantoja, Candace Ross, Adina Williams, and Megan Ung. Changing answer order can decrease mmlu accuracy. arXiv preprint:2406.19470, 2024. https://arxiv.org/abs/2406.19470

Suchin Gururangan, Ana Marasovic, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. Don’t stop pretraining: Adapt language models to domains and tasks. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault, editors, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, ACL 2020, Online, July 5–10, 2020, pages 8342–8360. Association for Computational Linguistics, 2020. doi: 10.18653/V1/2020.ACL-MAIN.740. https://doi.org/10.18653/v1/2020.acl-main.740

Momchil Hardalov, Todor Mihaylov, Dimitrina Zlatkova, Yoan Dinkov, Ivan Koychev, and Preslav Nakov. EXAMS: A multi-subject high school examinations dataset for cross-lingual and multilingual question answering. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu, editors, Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5427–5444, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.438. https://aclanthology.org/2020.emnlp-main.438

Thomas Hartvigsen, Saadia Gabriel, Hamid Palangi, Maarten Sap, Dipankar Ray, and Ece Kamar. Toxigen: A large-scale machine-generated dataset for adversarial and implicit hate speech detection. arXiv preprint arXiv:2203.09509, 2022.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3–7, 2021. OpenReview.net, 2021a. https://openreview.net/forum?id=d7KBjmI3GmQ

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Joaquin Vanschoren and Sai-Kit Yeung, editors, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, 2021b. https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican,

79

George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W Rae,
Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556,
2022.
Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Mia Xu Chen, Dehao Chen, HyoukJoong Lee, Jiquan
Ngiam, Quoc V. Le, Yonghui Wu, and Zhifeng Chen. Gpipe: Efficient training of giant neural networks using
pipeline parallelism, 2019.
Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing
Hu, Brian Fuller, Davide Testuginne, and Madian Khabsa. Llama guard: Llm-based input-output safeguard for
human-ai conversations. 2023.
Daphne Ippolito, Florian Tramer, Milad Nasr, Chiyuan Zhang, Matthew Jagielski, Katherine Lee, Christopher
Choquette Choo, and Nicholas Carlini. Preventing generation of verbatim memorization in language models
gives a false sense of privacy. In C. Maria Keet, Hung-Yi Lee, and Sina Zarrieß, editors, Proceedings of the 16th
International Natural Language Generation Conference, pages 28–53, Prague, Czechia, September 2023. Association
for Computational Linguistics. doi: 10.18653/v1/2023.inlg-main.3. https://aclanthology.org/2023.inlg-main.3.
Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Averaging weights
leads to wider optima and better generalization, 2019. https://arxiv.org/abs/1803.05407.
Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, and Joao Carreira. Perceiver: General
perception with iterative attention. arXiv preprint arXiv:2103.03206, 2021.
Meng Ji, Meng Ji, Pierrette Bouillon, and Mark Seligman. Cultural and Linguistic Bias of Neural Machine Translation
Technology, page 100–128. Studies in Natural Language Processing. Cambridge University Press, 2023.
Robin Jia and Percy Liang. Adversarial examples for evaluating reading comprehension systems. In Martha Palmer,
Rebecca Hwa, and Sebastian Riedel, editors, Proceedings of the 2017 Conference on Empirical Methods in Natural
Language Processing, pages 2021–2031, Copenhagen, Denmark, September 2017. Association for Computational
Linguistics. doi: 10.18653/v1/D17-1215. https://aclanthology.org/D17-1215.
Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas,
Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux,
Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b.
arXiv preprint arXiv:2310.06825, 2023.
Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh
Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint
arXiv:2401.04088, 2024.
Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with gpus. IEEE Transactions on Big
Data, 7(3):535–547, 2019.
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised
challenge dataset for reading comprehension. In Regina Barzilay and Min-Yen Kan, editors, Proceedings of the
55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–
1611, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1147.
https://aclanthology.org/P17-1147.
Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. Bag of tricks for efficient text classification.
In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics:
Volume 2, Short Papers, pages 427–431. Association for Computational Linguistics, April 2017.
Nal Kalchbrenner, Erich Elsen, Karen Simonyan, Seb Noury, Norman Casagrande, Edward Lockhart, Florian Stimberg,
Aaron Oord, Sander Dieleman, and Koray Kavukcuoglu. Efficient neural audio synthesis. In International Conference
on Machine Learning, pages 2410–2419. PMLR, 2018.
Gregory Kamradt. Llmtest_needleinahaystack. https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/
main/README.md, 2023.
Wonjune Kang, Yun Wang, Shun Zhang, Arthur Hinsvark, and Qing He. Multi-task learning for front-end text
processing in tts. In ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing
(ICASSP), pages 10796–10800, 2024. doi: 10.1109/ICASSP48485.2024.10446241.
80

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Aly M. Kassem, Omar Mahmoud, Niloofar Mireshghallah, Hyunwoo Kim, Yulia Tsvetkov, Yejin Choi, Sherif Saad, and Santu Rana. Alpaca against vicuna: Using llms to uncover memorization of llms, 2024. https://arxiv.org/abs/2403.04801.

Timo Kaufmann, Paul Weng, Viktor Bengs, and Eyke Hüllermeier. A survey of reinforcement learning from human feedback. arXiv preprint arXiv:2312.14925, 2023.

Aniruddha Kembhavi, Michael Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. ArXiv, abs/1603.07396, 2016. https://api.semanticscholar.org/CorpusID:2682274.

Eugene Kharitonov, Ann Lee, Adam Polyak, Yossi Adi, Jade Copet, Kushal Lakhotia, Tu-Anh Nguyen, Morgane Rivière, Abdelrahman Mohamed, Emmanuel Dupoux, et al. Text-free prosody-aware generative spoken language modeling. arXiv preprint arXiv:2109.03264, 2021.

Douwe Kiela, Max Bartolo, Yixin Nie, Divyansh Kaushik, Atticus Geiger, Zhengxuan Wu, Bertie Vidgen, Grusha Prasad, Amanpreet Singh, Pratik Ringshia, Zhiyi Ma, Tristan Thrush, Sebastian Riedel, Zeerak Waseem, Pontus Stenetorp, Robin Jia, Mohit Bansal, Christopher Potts, and Adina Williams. Dynabench: Rethinking benchmarking in NLP. In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou, editors, Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4110–4124, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.324. https://aclanthology.org/2021.naacl-main.324.

Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Muñoz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro von Werra, and Harm de Vries. The stack: 3 tb of permissively licensed source code, 2022. https://arxiv.org/abs/2211.15533.

Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. Mawps: A math word problem repository. In Proceedings of the 2016 conference of the north american chapter of the association for computational linguistics: human language technologies, pages 1152–1157, 2016.

Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models. Proceedings of Machine Learning and Systems, 5, 2023.

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In F. Pereira, C.J. Burges, L. Bottou, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems, volume 25. Curran Associates, Inc., 2012. https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention, 2023.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. RACE: Large-scale ReAding comprehension dataset from examinations. In Martha Palmer, Rebecca Hwa, and Sebastian Riedel, editors, Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 785–794, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1082. https://aclanthology.org/D17-1082.

Joel Lamy-Poirier. Breadth-first pipeline parallelism. Proceedings of Machine Learning and Systems, 5:48–67, 2023.

Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Leda Sari, Rashel Moritz, Mary Williamson, Vimal Manohar, Yossi Adi, Jay Mahadeokar, et al. Voicebox: Text-guided multilingual universal speech generation at scale. Advances in neural information processing systems, 36, 2024.

Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. Deduplicating training data makes language models better. arXiv preprint arXiv:2107.06499, 2021.

Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2struct: Screenshot parsing as pretraining for visual language understanding. In International Conference on Machine Learning, pages 18893–18912. PMLR, 2023.

Kevin Lee and Shubho Sengupta. Introducing the AI Research SuperCluster — Meta’s cutting-edge AI supercomputer for AI research, 2022. https://ai.meta.com/blog/ai-rsc/.

81

Kevin Lee, Adi Gangidi, and Mathew Oldham. Building meta’s genai infrastructure. 2024.
Jie Lei, Licheng Yu, Mohit Bansal, and Tamara L Berg. Tvqa: Localized, compositional video question answering. In EMNLP, 2018.
Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, and Luke Zettlemoyer. Base layers: Simplifying training of large, sparse models. In International Conference on Machine Learning, pages 6265–6274. PMLR, 2021.
Chen Li, Weiqi Wang, Jingcheng Hu, Yixuan Wei, Nanning Zheng, Han Hu, Zheng Zhang, and Houwen Peng. Common 7b language models already possess strong math capabilities. arXiv preprint arXiv:2403.04706, 2024a.
Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, Saurabh Garg, Rui Xin, Niklas Muennighoff, Reinhard Heckel, Jean Mercat, Mayee Chen, Suchin Gururangan, Mitchell Wortsman, Alon Albalak, Yonatan Bitton, Marianna Nezhurina, Amro Abbas, Cheng-Yu Hsieh, Dhruba Ghosh, Josh Gardner, Maciej Kilian, Hanlin Zhang, Rulin Shao, Sarah Pratt, Sunny Sanyal, Gabriel Ilharco, Giannis Daras, Kalyani Marathe, Aaron Gokaslan, Jieyu Zhang, Khyathi Chandu, Thao Nguyen, Igor Vasiljevic, Sham Kakade, Shuran Song, Sujay Sanghavi, Fartash Faghri, Sewoong Oh, Luke Zettlemoyer, Kyle Lo, Alaaeldin El-Nouby, Hadi Pouransari, Alexander Toshev, Stephanie Wang, Dirk Groeneveld, Luca Soldaini, Pang Wei Koh, Jenia Jitsev, Thomas Kollar, Alexandros G. Dimakis, Yair Carmon, Achal Dave, Ludwig Schmidt, and Vaishaal Shankar. Datacomp-lm: In search of the next generation of training sets for language models, 2024b. https://arxiv.org/abs/2406.11794.
KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023a.
Margaret Li, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A. Smith, and Luke Zettlemoyer. Branch-train-merge: Embarrassingly parallel training of expert language models, 2022. https://arxiv.org/abs/2208.03306.
Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li. Api-bank: A comprehensive benchmark for tool-augmented llms. arXiv preprint arXiv:2304.08244, 2023b.
Qintong Li, Leyang Cui, Xueliang Zhao, Lingpeng Kong, and Wei Bi. Gsm-plus: A comprehensive benchmark for evaluating the robustness of llms as mathematical problem solvers. arXiv preprint arXiv:2402.19255, 2024c.
Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel J. Orr, Lucia Zheng, Mert Yüksekgönül, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri S. Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta Koreeda. Holistic evaluation of language models. CoRR, abs/2211.09110, 2022. doi: 10.48550/ARXIV.2211.09110. https://doi.org/10.48550/arXiv.2211.09110.
Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.
Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023.
Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring attention with blockwise transformers for near-infinite context. arXiv preprint arXiv:2310.01889, 2023a.
Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2023b.
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023c.
Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. Advances in Neural Information Processing Systems, 36, 2024a.
Ruibo Liu, Jerry Wei, Fangyu Liu, Chenglei Si, Yanzhe Zhang, Jinmeng Rao, Steven Zheng, Daiyi Peng, Diyi Yang, Denny Zhou, and Andrew M. Dai. Best practices and lessons learned on synthetic data for language models. CoRR, abs/2404.07503, 2024b. doi: 10.48550/ARXIV.2404.07503. https://doi.org/10.48550/arXiv.2404.07503.
82


Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, and Junxian He. What makes good data for alignment? A comprehensive study of automatic data selection in instruction tuning, 2024c. https://arxiv.org/abs/2312.15685.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692, 2019a.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized BERT pretraining approach. CoRR, abs/1907.11692, 2019b. http://arxiv.org/abs/1907.11692.

Llama-Team. Meta llama guard 2. https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard2/MODEL_CARD.md, 2024.

Keming Lu, Hongyi Yuan, Zheng Yuan, Runji Lin, Junyang Lin, Chuanqi Tan, Chang Zhou, and Jingren Zhou. Instag: Instruction tagging for analyzing supervised fine-tuning of large language models, 2023.

Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8086–8098, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.556. https://aclanthology.org/2022.acl-long.556.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583, 2023.

Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In ACL, 2024.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36, 2024a.

Lovish Madaan, Aaditya K Singh, Rylan Schaeffer, Andrew Poulton, Sanmi Koyejo, Pontus Stenetorp, Sharan Narang, and Dieuwke Hupkes. Quantifying variance in evaluation benchmarks. arXiv preprint arXiv:2406.10229, 2024b.

Neelu Madan, Andreas Moegelmose, Rajat Modi, Yogesh S. Rawat, and Thomas B. Moeslund. Foundation models for video understanding: A survey, 2024.

Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens van der Maaten. Exploring the limits of weakly supervised pretraining. In Proceedings of the European Conference on Computer Vision (ECCV), September 2018.

Soumi Maiti, Yifan Peng, Shukjae Choi, Jee weon Jung, Xuankai Chang, and Shinji Watanabe. Voxtlm: unified decoder-only models for consolidating speech recognition/synthesis and speech/text continuation tasks, 2023.

Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Findings of the Association for Computational Linguistics: ACL 2022, pages 2263–2279, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-acl.177. https://aclanthology.org/2022.findings-acl.177.

Minesh Mathew, Dimosthenis Karatzas, R. Manmatha, and C. V. Jawahar. Docvqa: A dataset for vqa on document images. 2021 IEEE Winter Conference on Applications of Computer Vision (WACV), pages 2199–2208, 2020. https://api.semanticscholar.org/CorpusID:220280200.

Jeremy Baumgartner Matt Bowman. Meta open compute project, grand teton ai platform, 2022. https://engineering.fb.com/2022/10/18/open-source/ocp-summit-2022-grand-teton/.

Sachin Mehta, Mohammad Hossein Sekhavat, Qingqing Cao, Maxwell Horton, Yanzi Jin, Chenfan Sun, Iman Mirzadeh, Mahyar Najibi, Dmitry Belenko, Peter Zatloukal, et al. Openelm: An efficient language model family with open-source training and inference framework. arXiv preprint arXiv:2404.14619, 2024.

Dheeraj Mekala, Jason Weston, Jack Lanchantin, Roberta Raileanu, Maria Lomeli, Jingbo Shang, and Jane Dwivedi-Yu. Toolverifier: Generalization to new tools via self-verification. arXiv preprint arXiv:2402.14158, 2024.

83

Grégoire Mialon, Roberto Dessì, Maria Lomeli, Christoforos Nalmpantis, Ram Pasunuru, Roberta Raileanu, Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu, Asli Celikyilmaz, et al. Augmented language models: a survey. arXiv preprint arXiv:2302.07842, 2023a.

Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia: a benchmark for general AI assistants. arXiv preprint arXiv:2311.12983, 2023b.

Sabrina J. Mielke, Arthur Szlam, Y-Lan Boureau, and Emily Dinan. Linguistic calibration through metacognition: aligning dialogue agent responses with expected correctness. CoRR, abs/2012.14983, 2020. https://arxiv.org/abs/2012.14983

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii, editors, Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2381–2391, Brussels, Belgium, October–November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1260. https://aclanthology.org/D18-1260

Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781, 2013.

Swaroop Mishra, Daniel Khashabi, Chitta Baral, Yejin Choi, and Hannaneh Hajishirzi. Reframing instructional prompts to GPTk’s language. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Findings of the Association for Computational Linguistics: ACL 2022, pages 589–612, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-acl.50. https://aclanthology.org/2022.findings-acl.50

Arindam Mitra, Hamed Khanpour, Corby Rosset, and Ahmed Awadallah. Orca-math: Unlocking the potential of SLMs in grade school math. arXiv preprint arXiv:2402.14830, 2024.

Jean-Baptiste Mouret and Jeff Clune. Illuminating search spaces by mapping elites, 2015. https://arxiv.org/abs/1504.04909

Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng Xin Yong, Hailey Schoelkopf, et al. Crosslingual generalization through multitask finetuning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15991–16111, 2023.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. WebGPT: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, and Matei Zaharia. Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–15, 2021.

Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A. Feder Cooper, Daphne Ippolito, Christopher A. Choquette-Choo, Eric Wallace, Florian Tramèr, and Katherine Lee. Scalable extraction of training data from (production) language models. ArXiv, abs/2311.17035, 2023. https://api.semanticscholar.org/CorpusID:265466445

Tu Anh Nguyen, Benjamin Muller, Bokai Yu, Marta R. Costa-jussa, Maha Elbayad, Sravya Popuri Paul-Ambroise Duquenne, Robin Algayres, Ruslan Mavlyutov, Itai Gat, Gabriel Synnaeve, Juan Pino, Benoît Sagot, and Emmanuel Dupoux. Spirit-LM: Interleaved spoken and written language model. 2024.

Marta R. Costa-jussà NLLB Team, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang. No language left behind: Scaling human-centered machine translation. 2022.

OpenAI. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023a.

OpenAI. GPT-4 blog. https://openai.com/index/gpt-4-research/, 2023b.

OpenAI. simple-evals. https://github.com/openai/simple-evals, 2024.

84

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155, 2022.

Arka Pal, Deep Karkhanis, Samuel Dooley, Manley Roberts, Siddartha Naidu, and Colin White. Smaug: Fixing failure modes of preference optimisation with dpo-positive. arXiv preprint arXiv:2402.13228, 2024.

Liangming Pan, Michael Saxon, Wenda Xu, Deepak Nathani, Xinyi Wang, and William Yang Wang. Automatically correcting large language models: Surveying the Landscape of Diverse Automated Correction Strategies. Trans. Assoc. Comput. Linguistics, 12:484–506, 2024. doi: 10.1162/TACL_A_00660. https://doi.org/10.1162/tacl_a_00660.

Satadru Pan Pan, Theano Stavrinos, Yunqiao Zhang, Atul Sikaria, Pavel Zakharov, Abhinav Sharma, Shiva Shankar, Mike Shuey, Richard Wareing, Monika Gangapuram, Guanglei Cao, Christian Preseau, Pratap Singh, Kestutis Patiejunas, JR Tipton, Ethan Katz-Bassett, and Wyatt Lloyd. Facebook’s tectonic filesystem: Efficiency from exascale. In Proceedings of the 19th USENIX Conference on File and Storage Technologies, pages 217–231, 2021.

Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: an asr corpus based on public domain audio books. In 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 5206–5210. IEEE, 2015.

Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi, Nikita Nangia, Jason Phang, Angelica Chen, Vishakh Padmakumar, Johnny Ma, Jana Thompson, He He, and Samuel Bowman. QuALITY: Question answering with long input texts, yes! In Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz, editors, Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5336–5358, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.391. https://aclanthology.org/2022.naacl-main.391.

Richard Yuanzhe Pang, Weizhe Yuan, Kyunghyun Cho, He He, Sainbayar Sukhbaatar, and Jason Weston. Iterative reasoning preference optimization. arXiv preprint arXiv:2404.19733, 2024.

Aaron Parisi, Yao Zhao, and Noah Fiedel. Talm: Tool augmented language models. arXiv preprint arXiv:2205.12255, 2022.

Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. Gorilla: Large language model connected with massive apis. arXiv preprint arXiv:2305.15334, 2023.

Ed Pizzi, Sreya Dutta Roy, Sugosh Nagavara Ravindra, Priya Goyal, and Matthijs Douze. A self-supervised descriptor for image copy detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14532–14542, 2022.

B.T. Polyak. New stochastic approximation type procedures. Automation and Remote Control, 7(7), 1991.

Vineel Pratap, Qiantong Xu, Anuroop Sriram, Gabriel Synnaeve, and Ronan Collobert. Mls: A large-scale multilingual dataset for speech research. arXiv preprint arXiv:2012.03411, 2020.

Prokopis Prokopidis, Vassilis Papavassiliou, and Stelios Piperidis. Parallel global voices: a collection of multilingual corpora with citizen media stories. In Nicoletta Calzolari (Conference Chair), Khalid Choukri, Thierry Declerck, Sara Goggi, Marko Grobelnik, Bente Maegaard, Joseph Mariani, Helene Mazo, Asuncion Moreno, Jan Odijk, and Stelios Piperidis, editors, Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC 2016), Paris, France, may 2016. European Language Resources Association (ELRA). ISBN 978-2-9517408-9-1.

Viorica Pătrăucean, Lucas Smaira, Ankush Gupta, Adrià Recasens Continente, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Joseph Heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alex Frechette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, and João Carreira. Perception test: A diagnostic benchmark for multimodal video models. In NeurIPS, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine Mcleavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on

85

Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 28492–28518. PMLR, 23–29 Jul 2023. https://proceedings.mlr.press/v202/radford23a.html.

Jack W. Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, Eliza Rutherford, Tom Hennigan, Jacob Menick, Albin Cassirer, Richard Powell, George van den Driessche, Lisa Anne Hendricks, Maribeth Rauh, Po-Sen Huang, Amelia Glaese, Johannes Welbl, Sumanth Dathathri, Saffron Huang, Jonathan Uesato, John F. J. Mellor, Irina Higgins, Antonia Creswell, Nathan McAleese, Amy Wu, Erich Elsen, Siddhant M. Jayakumar, Elena Buchatskaya, David Budden, Esme Sutherland, Karen Simonyan, Michela Paganini, L. Sifre, Lena Martens, Xiang Lorraine Li, Adhiguna Kuncoro, Aida Nematzadeh, Elena Gribovskaya, Domenic Donato, Angeliki Lazaridou, Arthur Mensch, Jean-Baptiste Lespiau, Maria Tsimpoukelli, N. K. Grigorev, Doug Fritz, Thibault Sottiaux, Mantas Pajarskas, Tobias Pohlen, Zhitao Gong, Daniel Toyama, Cyprien de Masson d’Autume, Yujia Li, Tayfun Terzi, Vladimir Mikulik, Igor Babuschkin, Aidan Clark, Diego de Las Casas, Aurelia Guy, Chris Jones, James Bradbury, Matthew G. Johnson, Blake A. Hechtman, Laura Weidinger, Iason Gabriel, William S. Isaac, Edward Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem W. Ayoub, Jeff Stanway, L. L. Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. Scaling language models: Methods, analysis & insights from training gopher. ArXiv, abs/2112.11446, 2021. https://api.semanticscholar.org/CorpusID:245353475.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 2023.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models, 2020. https://arxiv.org/abs/1910.02054.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ questions for machine comprehension of text. In Jian Su, Kevin Duh, and Xavier Carreras, editors, Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392, Austin, Texas, November 2016. Association for Computational Linguistics. doi: 10.18653/v1/D16-1264. https://aclanthology.org/D16-1264.

Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don’t know: Unanswerable questions for SQuAD. In Iryna Gurevych and Yusuke Miyao, editors, Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 784–789, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-2124. https://aclanthology.org/P18-2124.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark, 2023. https://arxiv.org/abs/2311.12022.

Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, and Yuxiong He. Zero-offload: Democratizing billion-scale model training, 2021. https://arxiv.org/abs/2101.06840.

Joshua Robinson and David Wingate. Leveraging large language models for multiple choice question answering. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1–5, 2023. OpenReview.net, 2023. https://openreview.net/pdf?id=yKbprarjc5B.

Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, and Dirk Hovy. Xstest: A test suite for identifying exaggerated safety behaviours in large language models. arXiv preprint arXiv:2308.01263, 2023.

Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton-Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. Code llama: Open foundation models for code. CoRR, abs/2308.12950, 2023. doi: 10.48550/ARXIV.2308.12950. https://doi.org/10.48550/arXiv.2308.12950.

Paul K. Rubenstein, Chulayuth Asawaroengchai, Duc Dung Nguyen, Ankur Bapna, Zalán Borsos, Félix de Chaumont Quitry, Peter Chen, Dalia El Badawy, Wei Han, Eugene Kharitonov, Hannah Muckenhirn, Dirk Padfield,

86

James Qin, Danny Rozenberg, Tara Sainath, Johan Schalkwyk, Matt Sharifi, Michelle Tadmor Ramanovich, Marco Tagliasacchi, Alexandru Tudor, Mihajlo Velimirović, Damien Vincent, Jiahui Yu, Yongqiang Wang, Vicky Zayats, Neil Zeghidour, Yu Zhang, Zhishuai Zhang, Lukas Zilka, and Christian Frank. Audiopalm: A large language model that can speak and listen. 2023.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.

Mikayel Samvelyan, Sharath Chandra Raparthy, Andrei Lupu, Eric Hambro, Aram H. Markosyan, Manish Bhatt, Yuning Mao, Minqi Jiang, Jack Parker-Holder, Jakob Foerster, Tim Rocktäschel, and Roberta Raileanu. Rainbow teaming: Open-ended generation of diverse adversarial prompts, 2024. https://arxiv.org/abs/2402.16822.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019.

Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Teven Le Scao, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M Rush. Multitask prompted training enables zero-shot task generalization. In International Conference on Learning Representations, 2022. https://openreview.net/forum?id=9Vrb9D0WI4.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social IQa: Commonsense reasoning about social interactions. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4463–4473, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1454. https://aclanthology.org/D19-1454.

Beatrice Savoldi, Marco Gaido, Luisa Bentivogli, Matteo Negri, and Marco Turchi. Gender Bias in Machine Translation. Transactions of the Association for Computational Linguistics, 9:845–874, 08 2021. ISSN 2307-387X. doi: 10.1162/tacl_a_00401. https://doi.org/10.1162/tacl_a_00401.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36, 2024.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Seamless Communication, Loic Barrault, Yu-An Chung, Mariano Cora Meglioli, David Dale, Ning Dong, Paul-Ambroise Duquenne, Hady Elsahar, Hongyu Gong, Kevin Heffernan, John Hoffman, Christopher Klaiber, Pengwei Li, Daniel Licht, Jean Maillard, Alice Rakotoarison, Kaushik Ram Sadagopan, Guillaume Wenzek, Ethan Ye, Bapi Akula, Peng-Jen Chen, Naji El Hachem, Brian Ellis, Gabriel Mejia Gonzalez, Justin Haaheim, Prangthip Hansanti, Russ Howes, Bernie Huang, Min-Jae Hwang, Hirofumi Inaguma, Somya Jain, Elahe Kalbassi, Amanda Kallet, Ilia Kulikov, Janice Lam, Daniel Li, Xutai Ma, Ruslan Mavlyutov, Benjamin Peloquin, Mohamed Ramadan, Abinesh Ramakrishnan, Anna Sun, Kevin Tran, Tuan Tran, Igor Tufanov, Vish Vogeti, Carleigh Wood, Yilin Yang, Bokai Yu, Pierre Andrews, Can Balioglu, Marta R. Costa-jussà, Celebi Onur Maha Elbayad, Cynthia Gao, Francisco Guzmán, Justine Kao, Ann Lee, Alexandre Mourachko, Juan Pino, Sravya Popuri, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Paden Tomasello, Changhan Wang, Jeff Wang, and Skyler Wang. Seamlessm4t—massively multilingual & multimodal machine translation. ArXiv, 2023.

Uri Shaham, Maor Ivgi, Avia Efrat, Jonathan Berant, and Omer Levy. Zeroscrolls: A zero-shot benchmark for long text understanding. arXiv preprint arXiv:2305.14196, 2023.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, YK Li, Yu Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

87

Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. Language models are multilingual chain-of-thought reasoners, 2022. https://arxiv.org/abs/2210.03057.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism, 2019. http://arxiv.org/abs/1909.08053.

Aaditya Singh, Yusuf Kocyigit, Andrew Poulton, David Esiobu, Maria Lomeli, Gergely Szilvasy, and Dieuwke Hupkes. Evaluation data contamination in LLMs: how do we measure it and (when) does it matter? 2024.

Amanpreet Singh, Vivek Natarjan, Meet Shah, Yu Jiang, Xinlei Chen, Devi Parikh, and Marcus Rohrbach. Towards VQA models that can read. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8317–8326, 2019.

Snowflake. Snowflake Arctic: The Best LLM for Enterprise AI — Efficiently Intelligent, Truly Open blog. https://www.snowflake.com/blog/arctic-open-efficient-foundation-language-models-snowflake/, 2024.

Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Diffusion art or digital forgery? Investigating data replication in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6048–6058, 2023.

Venkat Krishna Srinivasan, Zhen Dong, Banghua Zhu, Brian Yu, Damon Mosk-Aoyama, Kurt Keutzer, Jiantao Jiao, and Jian Zhang. Nexusraven: a commercially-permissive language model for function calling. In NeurIPS 2023 Foundation Models for Decision Making Workshop, 2023.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568:127063, 2024.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei. Challenging BIG-bench tasks and whether chain-of-thought can solve them. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Findings of the Association for Computational Linguistics: ACL 2023, pages 13003–13051, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.824. https://aclanthology.org/2023.findings-acl.824.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149–4158, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1421. https://aclanthology.org/N19-1421.

Chunqiang Tang, Thawan Kooburat, Pradeep Venkatachalam, Akshay Chander, Zhe Wen, Aravind Narayanan, Patrick Dowell, and Robert Karl. Holistic Configuration Management at Facebook. In Proceedings of the 25th Symposium on Operating Systems Principles, pages 328–343, 2015.

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. 2024.

Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.

David Thiel. Identifying and eliminating CSAM in generative ML training data and models. Technical report, Stanford Internet Observatory, 2023.

Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, YaGuang Li, Hongrae Lee, Huaixiu Steven Zheng, Amin Ghafouri, Marcelo Menegali, Yanping Huang, Maxim Krikun, Dmitry Lepikhin, James Qin, Dehao Chen, Yuanzhong Xu, Zhifeng Chen, Adam Roberts, Maarten Bosma, Vincent Zhao, Yanqi Zhou, Chung-Ching Chang, Igor Krivokon, Will Rusch, Marc Pickett, Pranesh Srinivasan, Laichee Man, Kathleen Meier-Hellstern, Meredith Ringel Morris, Tulsee Doshi, Renelito Delos Santos, Toju Duke, Johnny Soraker, Ben Zevenbergen, Vinodkumar Prabhakaran, Mark Diaz, Ben Hutchinson, Kristen Olson, Alejandra Molina, Erin Hoffman-John, Josh Lee, Lora Aroyo, Ravi Rajakumar, Alena Butryna, Matthew Lamm, Viktoriya Kuzmina, Joe Fenton, Aaron Cohen, Rachel Bernstein, Ray Kurzweil, Blaise Aguera-Arcas, Claire Cui, Marian Croak, Ed Chi, and Quoc Le. Lamda: Language models for dialog applications, 2022. https://arxiv.org/abs/2201.08239.

88

Jörg Tiedemann. Parallel data, tools and interfaces in opus. In International Conference on Language Resources and Evaluation, 2012. https://api.semanticscholar.org/CorpusID:15453873.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process-and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 2017.

Bertie Vidgen, Adarsh Agrawal, Ahmed M Ahmed, Victor Akinwande, Namir Al-Nuaimi, Najla Alfaraj, Elie Alhajjar, Lora Aroyo, Trupti Bavalatti, Borhane Blili-Hamelin, et al. Introducing v0.5 of the ai safety benchmark from mlcommons. arXiv preprint arXiv:2404.12241, 2024.

Saranyan Vigraham and Benjamin Leonhardi. Maintaining large-scale ai capacity at meta. 2024.

Eric Wallace, Kai Xiao, Reimar Leike, Lilian Weng, Johannes Heidecke, and Alex Beutel. The instruction hierarchy: Training LLMs to prioritize privileged instructions, 2024. https://arxiv.org/abs/2404.13208.

Changhan Wang, Morgane Rivière, Ann Lee, Anne Wu, Chaitanya Talnikar, Daniel Haziza, Mary Williamson, Juan Pino, and Emmanuel Dupoux. Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation. arXiv preprint arXiv:2101.00390, 2021a.

Changhan Wang, Anne Wu, and Juan Pino. Covost 2 and massively multilingual speech-to-text translation. arXiv preprint arXiv:2007.10310, 2021b.

Haochun Wang, Sendong Zhao, Zewen Qiang, Bing Qin, and Ting Liu. Beyond the answers: Reviewing the rationality of multiple choice question answering for the evaluation of large language models. CoRR, abs/2402.01349, 2024a. doi: 10.48550/ARXIV.2402.01349. https://doi.org/10.48550/arXiv.2402.01349.

Jun Wang, Benjamin Rubinstein, and Trevor Cohn. Measuring and mitigating name biases in neural machine translation. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2576–2590, Dublin, Ireland, May 2022a. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.184. https://aclanthology.org/2022.acl-long.184.

Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Y Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce LLMs step-by-step without human annotations. CoRR, abs/2312.08935, 2023a.

Tianrui Wang, Long Zhou, Ziqiang Zhang, Yu Wu, Shujie Liu, Yashesh Gaur, Zhuo Chen, Jinyu Li, and Furu Wei. Viola: Unified codec language models for speech recognition, synthesis, and translation. 2023b.

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, et al. Super-naturalinstructions: Generalization via declarative instructions on 1600+ NLP tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5085–5109, 2022b.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. arXiv preprint arXiv:2406.01574, 2024b.

89

Zhiguo Wang, Wael Hamza, and Radu Florian. Bilateral multi-perspective matching for natural language sentences. arXiv preprint arXiv:1702.03814, 2017.

Lucas Weber, Elia Bruni, and Dieuwke Hupkes. Mind the instructions: a holistic evaluation of consistency and interactions in prompt-based learning. In Jing Jiang, David Reitter, and Shumin Deng, editors, Proceedings of the 27th Conference on Computational Natural Language Learning (CoNLL), pages 294–313, Singapore, December 2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023.conll-1.20. https://aclanthology.org/2023.conll-1.20.

Lucas Weber, Elia Bruni, and Dieuwke Hupkes. The icl consistency test. arXiv preprint arXiv:2312.04945, 2023b.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2022a.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022b. https://openreview.net/forum?id=yzkSU5zdwD.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022c.

Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, and Lingming Zhang. Magicoder: Empowering code generation with oss-instruct, 2024. https://arxiv.org/abs/2312.02120.

Sean Welleck, Ximing Lu, Peter West, Faeze Brahman, Tianxiao Shen, Daniel Khashabi, and Yejin Choi. Generating sequences by learning to self-correct. arXiv preprint arXiv:2211.00053, 2022.

Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, and Edouard Grave. Ccnet: Extracting high quality monolingual datasets from web crawl data, 2019. https://arxiv.org/abs/1911.00359.

Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time, 2022. https://arxiv.org/abs/2203.05482.

Chunyang Wu, Zhiping Xiu, Yangyang Shi, Ozlem Kalinli, Christian Fuegen, Thilo Koehler, and Qing He. Transformer-based acoustic modeling for streaming speech synthesis. In Interspeech, pages 146–150, 2021.

Haoyi Wu, Wenyang Hui, Yezeng Chen, Weiqi Wu, Kewei Tu, and Yi Zhou. Conic10k: A challenging math problem understanding and reasoning dataset, 2023. https://arxiv.org/abs/2311.05113.

Zhibiao Wu and Martha Palmer. Verb semantics and lexical selection. In ACL, 1994.

XAI. Open Release of Grok-1 blog. https://x.ai/blog/grok-os, 2024.

Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. 2024a.

Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models, 2024b.

Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In CVPR, 2021.

Yuxi Xie, Anirudh Goyal, Wenyue Zheng, Min-Yen Kan, Timothy P Lillicrap, Kenji Kawaguchi, and Michael Shieh. Monte carlo tree search boosts reasoning via iterative preference learning. arXiv preprint arXiv:2405.00451, 2024.

Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi Rungta, Karthik Abinav Sankararaman, Barlas Oguz, Madian Khabsa, Han Fang, Yashar Mehdad, Sharan Narang, Kshitiz Malik, Angela Fan, Shruti Bhosale, Sergey Edunov, Mike Lewis, Sinong Wang, and Hao Ma. Effective long-context scaling of foundation models. arXiv preprint arXiv:2309.16039, 2023.

Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying clip data. arXiv preprint arXiv:2309.16671, 2023.

90

Fanjia Yan, Huanzhi Mao, Charlie Cheng-Jie Ji, Tianjun Zhang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. Berkeley function calling leaderboard. https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html, 2024.

Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023a.

Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. 2023b.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, Chenliang Li, Yuanhong Xu, Hehong Chen, Junfeng Tian, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl: Modularization empowers large language models with multimodality. 2023.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023.

Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. In AAAI, 2019.

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv:2309.05653, 2023.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024a.

Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhu Chen. Mammoth2: Scaling instructions from the web. arXiv preprint arXiv:2405.03548, 2024b.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476–15488, 2022.

Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023.

Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Khai Hao, Xu Han, Zhen Leng Thai, Shuo Wang, Zhiyuan Liu, et al. ∞bench: Extending long context evaluation beyond 100k tokens. arXiv preprint arXiv:2402.13718, 2024.

Xinyu Zhang, Ian Colbert, Ken Kreutz-Delgado, and Srinjoy Das. Training deep neural networks with joint quantization and pruning of weights and activations, 2021.

Yuan Zhang, Jason Baldridge, and Luheng He. PAWS: Paraphrase adversaries from word scrambling. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 1298–1308, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1131. https://aclanthology.org/N19-1131.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023a. http://arxiv.org/abs/2303.18223.

Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li. Pytorch fsdp: Experiences on scaling fully sharded data parallel, 2023b.

Yue Zhao, Ishan Misra, Philipp Krähenbühl, and Rohit Girdhar. Learning video representations from large language models. In arXiv preprint arXiv:2212.04501, 2022.

Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving few-shot performance of language models. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International
91

Conference on Machine Learning, ICML 2021, 18–24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 12697–12706. PMLR, 2021. http://proceedings.mlr.press/v139/zhao21c.html.

Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. Large language models are not robust multiple choice selectors. CoRR, abs/2309.03882, 2023. doi: 10.48550/ARXIV.2309.03882. https://doi.org/10.48550/arXiv.2309.03882.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. arXiv preprint arXiv:2304.06364, 2023.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36, 2024.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M Dai, Quoc V Le, James Laudon, et al. Mixture-of-experts with expert choice routing. Advances in Neural Information Processing Systems, 35:7103–7114, 2022.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. 2023.

92

