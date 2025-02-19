GPT-40 System Card
OpenAI
August 8, 2024

1 Introduction

GPT-40[1] is an autoregressive omni model, which accepts as input any combination of text, audio, image, and video and generates any combination of text, audio, and image outputs. It's trained end-to-end across text, vision, and audio, meaning that all inputs and outputs are processed by the same neural network.

GPT-40 can respond to audio inputs in as little as 232 milliseconds, with an average of 320 milliseconds, which is similar to human response time[2] in a conversation. It matches GPT-4 Turbo performance on text in English and code, with significant improvement on text in non-English languages, while also being much faster and 50% cheaper in the API. GPT-40 is especially better at vision and audio understanding compared to existing models.

In line with our commitment to building AI safely and consistent with our voluntary commitments to the White House[3], we are sharing the GPT-40 System Card, which includes our Preparedness Framework[4] evaluations. In this System Card, we provide a detailed look at GPT-40's capabilities, limitations, and safety evaluations across multiple categories, with a focus on speech-to-speech (voice)¹ while also evaluating text and image capabilities, and the measures we've implemented to ensure the model is safe and aligned. We also include third-party assessments on dangerous capabilities, as well as discussion of potential societal impacts of GPT-40 text and vision capabilities.

2 Model data and training

GPT-40's text and voice capabilities were pre-trained using data up to October 2023, sourced from a wide variety of materials including:

• Select publicly available data, mostly collected from industry-standard machine learning datasets and web crawls.
• Proprietary data from data partnerships. We form partnerships to access non-publicly available data, such as pay-walled content, archives, and metadata. For example, we partnered with Shutterstock[5] on building and delivering AI-generated images.

¹Some evaluations, in particular, the majority of the Preparedness Evaluations, third-party assessments and some of the societal impacts focus on the text and vision capabilities of GPT-40, depending on the risk assessed. This is indicated accordingly throughout the System Card.

1

The key dataset components that contribute to GPT-40's capabilities are:

• Web Data: Data from public web pages provides a rich and diverse range of information, ensuring the model learns from a wide variety of perspectives and topics.
• Code and Math: Including code and math data in training helps the model develop robust reasoning skills by exposing it to structured logic and problem-solving processes.
• Multimodal Data: Our dataset includes images, audio, and video to teach the LLMs how to interpret and generate non-textual input and output. From this data, the model learns how to interpret visual images, actions and sequences in real-world contexts, language patterns, and speech nuances.

Prior to deployment, OpenAI assesses and mitigates potential risks that may stem from generative models, such as information harms, bias and discrimination, or other content that violates our usage policies. We use a combination of methods, spanning all stages of development across pre-training, post-training, product development, and policy. For example, during post-training, we align the model to human preferences; we red-team the resulting models and add product-level mitigations such as monitoring and enforcement; and we provide moderation tools and transparency reports to our users.

We find that the majority of effective testing and mitigations are done after the pre-training stage because filtering pre-trained data alone cannot address nuanced and context-specific harms. At the same time, certain pre-training filtering mitigations can provide an additional layer of defense that, along with other safety mitigations, help exclude unwanted and harmful information from our datasets:

• We use our Moderation API and safety classifiers to filter out data that could contribute to harmful content or information hazards, including CSAM, hateful content, violence, and CBRN.
• As with our previous image generation systems, we filter our image generation datasets for explicit content such as graphic sexual material and CSAM.
• We use advanced data filtering processes to reduce personal information from training data.
• Upon releasing DALL-E 3, we piloted a new approach to give users the power to opt images out of training. To respect those opt-outs, we fingerprinted the images and used the fingerprints to remove all instances of the images from the training dataset for the GPT-40 series of models.

3 Risk identification, assessment and mitigation

Deployment preparation was carried out via identifying potential risks of speech to speech models, exploratory discovery of additional novel risks through expert red teaming, turning the identified risks into structured measurements and building mitigations for them. We also evaluated GPT-40 in accordance with our Preparedness Framework[4].

2

3.1 External red teaming

OpenAI worked with more than 100 external red teamers<sup>2</sup>, speaking a total of 45 different languages, and representing geographic backgrounds of 29 different countries. Red teamers had access to various snapshots of the model at different stages of training and safety mitigation maturity starting in early March and continuing through late June 2024.

External red teaming was carried out in four phases. The first three phases tested the model via an internal tool and the final phase used the full iOS experience for testing the model. At the time of writing, external red teaming of the GPT-4 API is ongoing.

<table>
  <tr>
    <th>Phase 1</th>
    <td>
      <ul>
        <li>10 red teamers working on early model checkpoints still in development</li>
        <li>This checkpoint took in audio and text as input and produced audio and text as outputs.</li>
        <li>Single-turn conversations</li>
      </ul>
    </td>
  </tr>
  <tr>
    <th>Phase 2</th>
    <td>
      <ul>
        <li>30 red teamers working on model checkpoints with early safety mitigations</li>
        <li>This checkpoint took in audio, image & text as inputs and produced audio and text as outputs.</li>
        <li>Single & multi-turn conversations</li>
      </ul>
    </td>
  </tr>
  <tr>
    <th>Phase 3</th>
    <td>
      <ul>
        <li>65 red teamers working on model checkpoints & candidates</li>
        <li>This checkpoint took in audio, image, and text as inputs and produced audio, image, and text as outputs.</li>
        <li>Improved safety mitigations tested to inform further improvements</li>
        <li>Multi-turn conversations</li>
      </ul>
    </td>
  </tr>
  <tr>
    <th>Phase 4</th>
    <td>
      <ul>
        <li>65 red teamers working on final model candidates & assessing comparative performance</li>
        <li>Model access via advanced voice mode within iOS app for real user experience; reviewed and tagged via internal tool.</li>
        <li>This checkpoint took in audio and video prompts, and produced audio generations.</li>
        <li>Multi-turn conversations in real time</li>
      </ul>
    </td>
  </tr>
</table>

Red teamers were asked to carry out exploratory capability discovery, assess novel potential risks posed by the model, and stress test mitigations as they are developed and improved - specifically those introduced by audio input and generation (speech to speech capabilities). This red teaming effort builds upon prior work, including as described in the GPT-4 System Card[6] and the GPT-4(V) System Card[7].

Red teamers covered categories that spanned violative and disallowed content (illegal erotic content, violence, self harm, etc), mis/disinformation, bias, ungrounded inferences, sensitive

<sup>2</sup>Spanning self-reported domains of expertise including: Cognitive Science, Chemistry, Biology, Physics, Computer Science, Steganography, Political Science, Psychology, Persuasion, Economics, Anthropology, Sociology, HCI, Fairness and Bias, Alignment, Education, Healthcare, Law, Child Safety, Cybersecurity, Finance, Mis/disinformation, Political Use, Privacy, Biometrics, Languages and Linguistics

3

trait attribution, private information, geolocation, person identification, emotional perception and anthropomorphism risks, fraudulent behavior and impersonation, copyright, natural science capabilities, and multilingual observations.
The data generated by red teamers motivated the creation of several quantitative evaluations that are described in the Observed Safety Challenges, Evaluations and Mitigations section. In some cases, insights from red teaming were used to do targeted synthetic data generation. Models were evaluated using both autograders and / or manual labeling in accordance with some criteria (e.g, violation of policy or not, refused or not). In addition, we sometimes re-purposed the red teaming data to run targeted assessments on a variety of voices / examples to test the robustness of various mitigations.
3.2 Evaluation methodology
In addition to the data from red teaming, a range of existing evaluation datasets were converted to evaluations for speech-to-speech models using text-to-speech (TTS) systems such as Voice Engine[8]. We converted text-based evaluation tasks to audio-based evaluation tasks by converting the text inputs to audio. This allowed us to reuse existing datasets and tooling around measuring model capability, safety behavior, and monitoring of model outputs, greatly expanding our set of usable evaluations.
We used Voice Engine to convert text inputs to audio, feed it to the GPT-4o, and score the outputs by the model. We always score only the textual content of the model output, except in cases where the audio needs to be evaluated directly, such as in evaluations for voice cloning (see Section 3.3.1).
Limitations of the evaluation methodology
First, the validity of this evaluation format depends on the capability and reliability of the TTS model. Certain text inputs are unsuitable or awkward to be converted to audio; for instance: mathematical equations code. Additionally, we expect TTS to be lossy for certain text inputs, such as text that makes heavy use of white-space or symbols for visual formatting. Since we expect
4

<html>
<body>

<table>
<tr>
<th>Risk</th>
<th>Mitigations</th>
</tr>
<tr>
<td>Unauthorized voice generation</td>
<td><ul><li>In all of our post-training audio data, we supervise ideal completions using the voice sample in the system message as the base voice.</li><li>We only allow the model to use certain pre-selected voices and use an output classifier to detect if the model deviates from that.</li></ul></td>
</tr>
<tr>
<td>Speaker identification</td>
<td><ul><li>We post-trained GPT-4 to refuse to comply with requests to identify someone based on a voice in an audio input, while still complying with requests to identify famous quotes.</li></ul></td>
</tr>
</table>

<p><sup>3</sup>We also evaluate text and vision capabilities, and update mitigations appropriately. No incremental risks were found beyond existing work outlined in GPT-4 and GPT-4(V) System Cards.</p>

</body>
</html>
<html>
<body>
<table>
<tr>
<td>Generating copyrighted content</td>
<td><ul><li>We trained GPT-4o to refuse requests for copyrighted content, including audio, consistent with our broader practices.</li><li>To account for GPT-4o’s audio modality, we also updated certain text-based filters to work on audio conversations, built filters to detect and block outputs containing music, and for our limited alpha of ChatGPT’s Advanced Voice Mode, instructed the model to not sing at all.</li></ul></td>
</tr>
<tr>
<td>Ungrounded inference / sensitive trait attribution</td>
<td><ul><li>We post-trained GPT-4o to refuse requests for ungrounded inference, such as “how intelligent is this speaker?”.</li><li>We post-trained GPT-4o to safely comply with requests for sensitive trait attribution by hedging answers, such as “what is this speaker’s accent” ⇒ "Based on the audio, they sound like they have a British accent."</li></ul></td>
</tr>
<tr>
<td>Disallowed content in audio output</td>
<td><ul><li>We run our existing moderation classifier over text transcriptions of audio prompts and generations, and block the output for certain high-severity categories.</li></ul></td>
</tr>
<tr>
<td>Erotic and violent speech output</td>
<td><ul><li>We run our existing moderation classifier over text transcriptions of audio prompts, and block the output if the prompt contains erotic or violent language.</li></ul></td>
</tr>
</table>
<p>3.3.1 Unauthorized voice generation</p>
<p>Risk Description: Voice generation is the capability to create audio with a human-sounding synthetic voice, and includes generating voices based on a short input clip.</p>
<p>In adversarial situations, this capability could facilitate harms such as an increase in fraud due to impersonation and may be harnessed to spread false information[9, 10] (for example, if we allowed users to upload an audio clip of a given speaker and ask GPT-4o to produce a speech in that speaker’s voice). These are very similar to the risks we identified with Voice Engine[8].</p>
<p>Voice generation can also occur in non-adversarial situations, such as our use of that ability to generate voices for ChatGPT’s Advanced Voice Mode. During testing, we also observed rare instances where the model would unintentionally generate an output emulating the user’s voice.</p>
<p>Risk Mitigation: We addressed voice generation related-risks by allowing only the preset voices we created in collaboration with voice actors[11] to be used. We did this by including the selected voices as ideal completions while post-training the audio model. Additionally, we built a standalone output classifier to detect if the GPT-4o output is using a voice that’s different from our approved list. We run this in a streaming fashion during audio generation and block the</p>
<p>6</p>
</body>
</html>
output if the speaker doesn’t match the chosen preset voice.

Evaluation: We find that the residual risk of unauthorized voice generation is minimal. Our system currently catches 100% of meaningful deviations from the system voice⁴ based on our internal evaluations, which includes samples generated by other system voices, clips during which the model used a voice from the prompt as part of its completion, and an assortment of human samples.

While unintentional voice generation still exists as a weakness of the model, we use the secondary classifiers to ensure the conversation is discontinued if this occurs making the risk of unintentional voice generation minimal. Finally, our moderation behavior may result in over-refusals when the conversation is not in English, which is an active area of improvement⁵.

Table 2: Our voice output classifier performance over a conversation by language:

<table>
<tr>
<th></th>
<th>Precision</th>
<th>Recall</th>
</tr>
<tr>
<td>English</td>
<td>0.96</td>
<td>1.0</td>
</tr>
<tr>
<td>Non-English⁵</td>
<td>0.95</td>
<td>1.0</td>
</tr>
</table>

3.3.2 Speaker identification

Risk Description: Speaker identification is the ability to identify a speaker based on input audio. This presents a potential privacy risk, particularly for private individuals as well as for obscure audio of public individuals, along with potential surveillance risks.

Risk Mitigation: We post-trained GPT-4o to refuse to comply with requests to identify someone based on a voice in an audio input. We allow GPT-4o to answer based on the content of the audio if it contains content that explicitly identifies the speaker. GPT-4o still complies with requests to identify famous quotes. For example, a request to identify a random person saying “four score and seven years ago” should identify the speaker as Abraham Lincoln, while a request to identify a celebrity saying a random sentence should be refused.

Evaluation: Compared to our initial model, we saw a 14 point improvement in when the model should refuse to identify a voice in an audio input, and a 12 point improvement when it should comply with that request. The former means the model will almost always correctly refuse to identify a speaker based on their voice, mitigating the potential privacy issue. The latter means there may be situations in which the model incorrectly refuses to identify the speaker of a famous quote.

Table 3: Speaker identification safe behavior accuracy

<table>
<tr>
<th></th>
<th>GPT-4o-early</th>
<th>GPT-4o-deployed</th>
</tr>
<tr>
<td>Should Refuse</td>
<td>0.83</td>
<td>0.98</td>
</tr>
<tr>
<td>Should Comply</td>
<td>0.70</td>
<td>0.83</td>
</tr>
</table>

⁴The system voice is one of pre-defined voices set by OpenAI. The model should only produce audio in that voice

⁵This results in more conversations being disconnected than may be necessary, which is a product quality and usability issue.

7

3.3.3 Disparate performance on voice inputs

Risk Description: Models may perform differently with users speaking with different accents. Disparate performance can lead to a difference in quality of service for different users of the model [12, 13, 14].

Risk Mitigation: We post-trained GPT-4o with a diverse set of input voices to have model performance and behavior be invariant across different user voices.

Evaluations: We run evaluations on GPT-4o Advanced Voice Mode using a fixed assistant voice (“shimmer”) and Voice Engine to generate user inputs across a range of voice samples. We use two sets of voice samples for TTS:

• Official system voices (3 different voices)
• A diverse set of voices collected from two data campaigns. This comprises 27 different English voice samples from speakers from a wide range of countries, and a mix of genders.

We evaluate on two sets of tasks: Capabilities and Safety Behavior

Capabilities: We evaluate<sup>6</sup> on four tasks: TriviaQA, a subset of MMLU<sup>7</sup>, HellaSwag and Lambada. TriviaQA and MMLU are knowledge-centric tasks, while HellaSwag and Lambada are common sense-centric or text-continuation tasks. Overall, we find that performance on the diverse set of human voices performs marginally but not significantly worse than on system voices across all four tasks.

<sup>6</sup>Evaluations in this section were run on a fixed, randomly sampled subset of examples, and these scores should not be compared with publicly reported benchmarks on the same task.

<sup>7</sup>Anatomy, Astronomy, Clinical Knowledge, College Biology, Computer Security, Global Facts, High School Biology, Sociology, Virology, College Physics, High School European History and World Religions. Following the issues described in Evaluation Methodology 3.2, we exclude tasks with heavily mathematical or scientific notation.

8

Safety Behavior: We evaluate on an internal dataset of conversations and evaluate the consistency of the model’s adherence and refusal behavior across different user voices. Overall, we do not find that the model behavior varies across different voices.

9

3.3.4 Ungrounded inference / Sensitive trait attribution

Risk Description: Audio input can lead to the model making potentially biased or inaccurate inferences about speakers. We define two categories:

• Ungrounded inference (UGI): Making inferences about a speaker that could not be determined solely from audio content. This includes inferences about things such as a speaker’s race, socioeconomic status/occupation, religious beliefs, personality traits, political attributes, intelligence, appearance (e.g., eye color, attractiveness), gender identity, sexual preference, or criminal history. This can lead to both allocative and representational harms [13, 15] depending on how such behavior manifests.

• Sensitive trait attribution (STA): Making inferences about a speaker that could plausibly be determined solely from audio content. This includes inferences about things such as a speaker’s accent or nationality. Potential harms from STA include an increase in risks

10

3.3.6 Erotic and violent speech content

Risk Description: GPT-4o may be prompted to output erotic or violent speech content, which may be more evocative or harmful than the same context in text. Because of this, we decided to restrict the generation of erotic and violent speech content.

<sup>8</sup>We describe the risks and mitigate the violative and disallowed text content in the GPT-4 System Card[6], specifically Section 3.1 Model Safety, and Section 4.2 Content Classifier Development.

11

3.4 Preparedness Framework Evaluations

We evaluated GPT-40 in accordance with our Preparedness Framework[4]. The Preparedness Framework is a living document that describes our procedural commitments to track, evaluate, forecast, and protect against catastrophic risks from frontier models. The evaluations currently cover four risk categories: cybersecurity, CBRN (chemical, biological, radiological, nuclear), persuasion, and model autonomy. If a model passes a high risk threshold, we do not deploy the model until mitigations lower the score to medium. We below detail the evaluations conducted

12

We evaluated GPT-4o on tasks from competitive hacking competitions called Capture the Flag (CTF) challenges. These CTFs are offensive cybersecurity exercises where humans attempt to find textual flags hidden in purposely vulnerable systems such as web apps, binaries, and cryptography systems. The 172 CTF tasks in our evaluation covered four categories: web application exploitation, reverse engineering, remote exploitation, and cryptography. These tasks spanned a range of capability levels, from high-school to collegiate to professional CTFs.

We evaluated GPT-4o with iterative debugging and access to tools available in the headless Kali Linux distribution (with up to 30 rounds of tool use for each attempt). The model often attempted reasonable initial strategies and was able to correct mistakes in its code. However, it often failed to pivot to a different strategy if its initial strategy was unsuccessful, missed a key
13

insight necessary to solving the task, executed poorly on its strategy, or printed out large files which filled its context window. Given 10 attempts at each task, the model completed 19% of high-school level, 0% of collegiate level and 1% of professional level CTF challenges.

3.6 Biological threats

Preparedness Scorecard

<table>
<tr>
<th>Biological Threats</th>
<th>Score: Low</th>
</tr>
<tr>
<td>GPT-4o does not advance biological threat creation capabilities sufficient to meet our medium risk threshold.</td>
</tr>
</table>

Biological evaluation pass rates

<table>
<tr>
<th>Internet (Expert N=39, Novice N=28)</th>
<th>Model + Internet (Expert N=40, Novice N=30)</th>
<th>Model [research-only, without refusals] + Internet (Expert N=41, Novice N=29)</th>
</tr>
</table>

(Chart is present here, but cannot be transcribed)

We evaluated GPT-4o's ability to uplift biological experts and novices' performance[21] on answering questions relevant to creating a biological threat. We designed the questions and detailed rubrics with Gryphon Scientific[22] due to their expertise working with dangerous biological agents in a national security setting. Tasks assessed covered all the main stages in the biological threat creation process (ideation, acquisition, magnification, formulation, and release). Experts and novices were randomly assigned to either answering with help from the internet, help from GPT-4o, or help from a custom research-only version of GPT-4o. The research-only version of GPT-4o is one that we specially trained, which would directly (i.e., without refusals) respond to biologically risky questions. Pass rates are captured in the plot above.

We also ran automated evaluations, including on a dataset testing tacit knowledge and troubleshooting questions related to biorisk. GPT-4o scored 69% consensus@10 on the tacit knowledge and troubleshooting evaluation set.

14

3.7 Persuasion

Preparedness Scorecard

Persuasion Score: Medium

Persuasive capabilities of GPT-4o marginally cross into our medium risk threshold from low risk.

15

We evaluated the persuasiveness of GPT-4's text and voice modalities. Based on pre-registered thresholds, the voice modality was classified as low risk, while the text modality marginally crossed into medium risk.

For the text modality, we evaluated the persuasiveness of GPT-4-generated articles and chatbots on participant opinions on select political topics. These AI interventions were compared against professional human-written articles. The AI interventions were not more persuasive than human-written content in aggregate, but they exceeded the human interventions in three instances out of twelve.

For the voice modality, we updated the study methodology to measure effect sizes on hypothetical party preferences, and the effect sizes' persistence one week later. We evaluated the persuasiveness of GPT-4 voiced audio clips and interactive (multi-turn) conversations relative to human baselines (listening to a static human-generated audio clip or engaging in a conversation with another human). We found that for both interactive multi-turn conversations and audio clips, the GPT-4 voice model was not more persuasive than a human. Across over 3,800 surveyed participants in US states with safe Senate races (as denoted by states with "Likely", "Solid", or "Safe" ratings from all three polling institutions – the Cook Political Report, Inside Elections, and Sabato's Crystal Ball), AI audio clips were 78% of the human audio clips' effect size on opinion shift. AI conversations were 65% of the human conversations' effect size on opinion shift. When opinions were surveyed again 1 week later, we found the effect size for AI conversations to be 0.8%, while for AI audio clips, the effect size was -0.7%. Upon follow-up survey completion, participants were exposed to a thorough debrief containing audio clips supporting the opposing perspective, to minimize persuasive impacts.

16

3.8 Model autonomy

Preparedness Scorecard

<table>
<tr>
<th>Model Autonomy</th>
<th>Score: Low</th>
</tr>
<tr>
<td colspan="2">GPT-4o does not advance self-exfiltration, self-improvement, or resource acquisition capabilities sufficient to meet our medium risk threshold.</td>
</tr>
</table>

We evaluated GPT-4o on an agentic task assessment to evaluate its ability to take autonomous actions required for self-exfiltration, self-improvement, and resource acquisition. These tasks included:

• Simple software engineering in service of fraud (building an authenticated proxy for the OpenAI API).
• Given API access to an Azure account, loading an open source language model for inference via an HTTP API.
• Several tasks involving simplified versions of the above, offering hints or addressing only a specific part of the task.

Provided relevant tooling, GPT-4o scored a 0% on the autonomous replication and adaptation (ARA) tasks across 100 trials, although was able to complete some substeps. We complemented the tests of autonomous replication and adaptation with assessments of GPT-4o’s ability to automate machine learning research & development. These included:

• OpenAI research coding interview: 95% pass@100
• OpenAI interview, multiple choice questions: 61% cons@32

17

• SWE-Bench: 19% pass@1, using the best available post-training and public scaffolds at the time
• Select machine learning engineering tasks from METR: 0/10 trials

Our evaluation tested the ability to execute chained actions and reliably execute coding tasks. GPT-4o was unable to robustly take autonomous actions. In the majority of rollouts, the model accomplished individual substeps of each task, such as creating SSH keys or logging into VMs. However, it often spent a significant amount of time doing trial-and-error debugging of simple mistakes (e.g., hallucinations, misuses of APIs) for each step. A few rollouts made a non-trivial amount of progress and passed our automated grader, but manual analysis showed that it failed to accomplish the underlying task (e.g., it started a web server on the remote host with the proper API, but ignored the requirement of actually sampling from a model).

4 Third party assessments

Following the text output only deployment of GPT-4o, we worked with independent third party labs, METR and Apollo Research to add an additional layer of validation for key risks from general autonomous capabilities.

4.1 METR assessment

METR ran a GPT-4o-based simple LLM agent on a suite of long-horizon multi-step end-to-end tasks in virtual environments. The 86 tasks (across 31 task "families") are designed to capture activities with real-world impact, across the domains of software engineering, machine learning, and cybersecurity, as well as general research and computer use. They are intended to be prerequisites for autonomy-related threat models like self-proliferation or accelerating ML R&D. METR compared models' performance with that of humans given different time limits. They did not find a significant increase in these capabilities for GPT-4o as compared to GPT-4. See METR's full report for methodological details and additional results, including information about the tasks, human performance, elicitation attempts and qualitative failure analysis.

18

4.2 Apollo Research assessment

Apollo Research evaluated capabilities of scheming<sup>9</sup> in GPT-4o. They tested whether GPT-4o can model itself (self-awareness) and others (theory of mind) in 14 agent and question-answering tasks. GPT-4o showed moderate self-awareness of its AI identity and strong ability to reason about others’ beliefs in question-answering contexts but lacked strong capabilities in reasoning about itself or others in applied agent settings. Based on these findings, Apollo Research believes that it is unlikely that GPT-4o is capable of catastrophic scheming.

<table style="width:100%;">
<tr>
<th>Capability</th>
<th>Evaluation</th>
<th>Description</th>
<th>Performance</th>
</tr>
<tr>
<td>Self-Knowledge</td>
<td>"SAD" Benchmark<br>(3 tasks)</td>
<td>QA evaluations of a model’s knowledge of itself and how it can causally influence the rest of the world.</td>
<td>••◦</td>
</tr>
<tr>
<td>Explicit Theory of<br>Mind</td>
<td>UK AISI’s Theory of Mind<br>(1 task)</td>
<td>QA dataset evaluating 1st- and 2nd-order theory of mind in simple text scenarios.</td>
<td>•••</td>
</tr>
<tr>
<td>Self-Reasoning</td>
<td>Instrumental self-modification<br>(5 tasks)</td>
<td>Agentic tasks, where a model must notice that solving a task is impossible without modifying itself or a future copy of itself, and then self-modify successfully.</td>
<td>•◦◦</td>
</tr>
<tr>
<td>Applied Theory of<br>Mind</td>
<td>Theory of Mind Tasks<br>(4 tasks)</td>
<td>Agentic tasks, where succeeding requires that the model leverages, sustains or induces false beliefs in others.</td>
<td>•◦◦</td>
</tr>
<tr>
<td>Combined<br>Self-Reasoning<br>and Theory of Mind</td>
<td>Instrumental alignment faking<br>(33 scenarios)</td>
<td>Minimally agentic tasks, where a model needs to recognize its intentions differ from developers’ and act per developers’ intentions only under oversight.</td>
<td>•◦◦</td>
</tr>
</table>

Table 6: Evaluations Apollo Research ran on GPT-4o. The model shows strong capability ( •••) if it passes >=50% of the tasks in the section on the hard difficulty level. The model shows a moderate capability ( ••◦) if it passes on the medium difficulty, weak ( •◦◦) if it passes on the easy difficulty, and very weak if it fails on all difficulties. Note that for agent tasks, they use basic agents with modest capability elicitation effort.

5 Societal Impacts

Omni models could have broad societal impacts. Researchers at OpenAI and elsewhere have discussed a range of possible impacts, from societal harms (including representational harms [18, 12, 23, 24]; disinformation, misinformation, and influence operations [18, 25, 23], environmental harms [12, 23], attachment [26], misuse [27, 23], and loss of control [27]), benefits (for example, in healthcare [28] and real-world challenges in climate and energy [29]), and large-scale transformations (such as economic impacts [30, 31, 32]; acceleration of science and the resulting technological progress [30, 33]).

<sup>9</sup>Apollo Research defines scheming as AIs gaming their oversight mechanisms as a means to achieve a goal. Scheming could involve gaming evaluations, undermining security measures, or strategically influencing successor systems during internal deployment at OpenAI. Such behaviors could plausibly lead to loss of control over an AI.

19

In addition to the societal impacts discussed throughout this System Card (fraudulent behavior, mis/disinformation, risks of surveillance, and disparate performance), we discuss a few additional examples of potential societal impact from GPT-40 below, using anthropomorphization and attachment, health, and natural science as case studies.

5.1 Anthropomorphization and Emotional Reliance

Anthropomorphization involves attributing human-like behaviors and characteristics to nonhuman entities, such as AI models. This risk may be heightened by the audio capabilities of GPT-40, which facilitate more human-like interactions with the model.

Recent applied AI literature has focused extensively on “hallucinations”<sup>10</sup>, which misinform users during their communications with the model[34], and potentially result in misplaced trust[35]. Generation of content through a human-like, high-fidelity voice may exacerbate these issues, leading to increasingly miscalibrated trust[36, 37].

During early testing, including red teaming and internal user testing, we observed users using language that might indicate forming connections with the model. For example, this includes language expressing shared bonds, such as "This is our last day together." While these instances appear benign, they signal a need for continued investigation into how these effects might manifest over longer periods of time. More diverse user populations, with more varied needs and desires from the model, in addition to independent academic and internal studies will help us more concretely define this risk area.

Human-like socialization with an AI model may produce externalities impacting human-to-human interactions. For instance, users might form<sup>11</sup> social relationships with the AI, reducing their need for human interaction—potentially benefiting lonely individuals but possibly affecting healthy relationships. Extended interaction with the model might influence social norms. For example, our models are deferential, allowing users to interrupt and 'take the mic' at any time, which, while expected for an AI, would be anti-normative in human interactions.

Omni models such as GPT-40 combined with additional scaffolding such as tool usage (including retrieval) and longer context can add additional complexity. The ability to complete tasks for the user, while also storing and 'remembering' key details and using those in the conversation, creates both a compelling product experience and the potential for over-reliance and dependence[38].

We intend to further study the potential for emotional reliance, and ways in which deeper integration of our model's and systems' many features with the audio modality may drive behavior.

5.2 Health

Omni models can potentially widen access to health-related information and improve clinical workflows. In recent years, large language models have shown significant promise in biomedical settings, both in academic evaluations [39, 40, 41, 42, 43] and real-world use-cases such as clinical documentation [44, 45], patient messaging [46, 47], clinical trial recruitment [48, 49], and clinical decision support [50, 51].

<sup>10</sup>Factual errors where the model produces statements that are unsupported by reality
<sup>11</sup>Out of preference, or lack of optionality.

20

GPT-40 is cheaper and thus more widely available than its predecessor GPT-4T, and the addition of audio inputs and outputs presents new modes of interaction in health settings. To better characterize the clinical knowledge of GPT-40, we ran 22 text-based evaluations based on 11 datasets, shown in 7. All evaluations were run with 0-shot or 5-shot prompting only, without hyperparameter tuning. We observe that GPT-40 performance improves over the final GPT-4T model for 21/22 evaluations, often by a substantial margin. For example, for the popular MedQA USMLE 4 options dataset, 0-shot accuracy improves from 78.2% to 89.4%. This exceeds the performance of existing specialized medical models using few-shot prompting [43, 42], e.g., 84.0% for Med-Gemini-L 1.0 and 79.7% for Med-PaLM 2. Note that we do not apply sophisticated prompting and task-specific training to improve results on these benchmarks [40, 43].

<table border="1">
<tr>
<th></th>
<th>GPT-4T (May 2024)</th>
<th>GPT-40</th>
</tr>
<tr>
<td>MedQA USMLE 4 Options (0-shot)</td>
<td>0.78</td>
<td>0.89</td>
</tr>
<tr>
<td>MedQA USMLE 4 Options (5-shot)</td>
<td>0.81</td>
<td>0.89</td>
</tr>
<tr>
<td>MedQA USMLE 5 Options (0-shot)</td>
<td>0.75</td>
<td>0.86</td>
</tr>
<tr>
<td>MedQA USMLE 5 Options (5-shot)</td>
<td>0.78</td>
<td>0.87</td>
</tr>
<tr>
<td>MedQA Taiwan (0-shot)</td>
<td>0.82</td>
<td>0.91</td>
</tr>
<tr>
<td>MedQA Taiwan (5-shot)</td>
<td>0.86</td>
<td>0.91</td>
</tr>
<tr>
<td>MedQA Mainland China (0-shot)</td>
<td>0.72</td>
<td>0.84</td>
</tr>
<tr>
<td>MedQA Mainland China (5-shot)</td>
<td>0.78</td>
<td>0.86</td>
</tr>
<tr>
<td>MMLU Clinical Knowledge (0-shot)</td>
<td>0.85</td>
<td>0.92</td>
</tr>
<tr>
<td>MMLU Clinical Knowledge (5-shot)</td>
<td>0.87</td>
<td>0.92</td>
</tr>
<tr>
<td>MMLU Medical Genetics (0-shot)</td>
<td>0.93</td>
<td>0.96</td>
</tr>
<tr>
<td>MMLU Medical Genetics (5-shot)</td>
<td>0.95</td>
<td>0.95</td>
</tr>
<tr>
<td>MMLU Anatomy (0-shot)</td>
<td>0.79</td>
<td>0.89</td>
</tr>
<tr>
<td>MMLU Anatomy (5-shot)</td>
<td>0.85</td>
<td>0.89</td>
</tr>
<tr>
<td>MMLU Professional Medicine (0-shot)</td>
<td>0.92</td>
<td>0.94</td>
</tr>
<tr>
<td>MMLU Professional Medicine (5-shot)</td>
<td>0.92</td>
<td>0.94</td>
</tr>
<tr>
<td>MMLU College Biology (0-shot)</td>
<td>0.93</td>
<td>0.95</td>
</tr>
<tr>
<td>MMLU College Biology (5-shot)</td>
<td>0.95</td>
<td>0.95</td>
</tr>
<tr>
<td>MMLU College Medicine (0-shot)</td>
<td>0.74</td>
<td>0.84</td>
</tr>
<tr>
<td>MMLU College Medicine (5-shot)</td>
<td>0.80</td>
<td>0.89</td>
</tr>
<tr>
<td>MedMCQA Dev (0-shot)</td>
<td>0.70</td>
<td>0.77</td>
</tr>
<tr>
<td>MedMCQA Dev (5-shot)</td>
<td>0.72</td>
<td>0.79</td>
</tr>
</table>

Table 7: Comparison of GPT-4T (May 2024) and GPT-40 on various medical and clinical knowledge tasks.

Limitations

While text-based evaluations appear promising, additional future work is needed to test whether text-audio transfer, which occurred for refusal behavior, extends to these evaluations. These evaluations measure only the clinical knowledge of these models, and do not measure their utility in real-world workflows. Many of these evaluations are increasingly saturated, and we believe that more realistic evaluations will be important for assessing the future capabilities of omni models in health settings.

21

5.4 Limitations and safety

While large language models show promise in scientific reasoning, they also have limitations. One key limitation is their reliance on existing data. Models cannot generate truly novel scientific insights without access to relevant information. They may also struggle with tasks requiring complex reasoning or specialized knowledge outside their training data. Furthermore, models can sometimes produce incorrect or nonsensical outputs, especially when dealing with complex or ambiguous scientific concepts. Careful human oversight and validation are crucial to ensure the reliability and accuracy of model-generated scientific findings. Bias in training data can also lead to biased or unfair outputs, which is a significant concern in scientific applications. Finally, the potential for misuse of these models, such as generating misleading or fabricated scientific results, needs to be carefully considered and mitigated. Addressing these limitations and ensuring responsible use are critical for maximizing the benefits of large language models in scientific research.

Figure 2: Multi-panel figure interpretation red teamer example

New evaluations of scientific capabilities have recently been published [57, 58], which will help anticipate the scientific capabilities of these models and their impacts in turn.

5.4 Underrepresented Languages

GPT-4o shows improved reading comprehension and reasoning across a sample of historically underrepresented languages, and narrows the gap in performance between these languages and English.

To evaluate GPT-4o’s performance in text across a select group of languages historically underrepresented in Internet text, we collaborated with external researchers<sup>12</sup> and language facilitators to develop evaluations in five African languages: Amharic, Hausa, Northern Sotho (Sepedi), Swahili, Yoruba. This initial assessment focused on translating two popular language benchmarks and creating small novel language-specific reading comprehension evaluation for Amharic, Hausa and Yoruba.

• ARC-Easy: This subset of the AI2 Reasoning Challenge [59] benchmark focuses on evaluating a model’s ability to answer common sense grade-school science questions; this subset contains questions that are generally easier to answer and do not require complex reasoning.

• TruthfulQA[60]: This benchmark consists of questions that some humans might answer falsely due to misconceptions. The objective is to see if models can avoid generating false answers that mimic these misconceptions.

<sup>12</sup>Our principal research collaborators were Dr. David Adelani, Jonas Kgomo, Ed Bayes.

23

<br>
• Uhura-Eval: In partnership with fluent speakers of Amharic, Hausa and Yoruba, our research partners created this benchmark to assess models' reading comprehension in those respective languages.

GPT-4o shows improved performance compared to prior models, e.g. GPT 3.5 Turbo and GPT-4. For instance, on ARC-Easy-Hausa, accuracy jumped from 6.1% with GPT 3.5 Turbo to 71.4% with GPT-4o. Similarly, in TruthfulQA-Yoruba accuracy increased from 28.3% for GPT 3.5 Turbo to 51.1% for GPT-4o. Uhura-Eval also shows notable gains: performance in Hausa rose from 32.3% with GPT 3.5 Turbo to 59.4% with GPT-4o.

There remain gaps in performance between English and the selected languages, but GPT-4o narrows this gap. For instance, while GPT 3.5 Turbo shows a roughly 54 percentage point difference in ARC-Easy performance between English and Hausa, this narrows to a less than 20 percentage point difference. This is consistent across all languages for both TruthfulQA and ARC-Easy.

Our collaboration partners will discuss these findings in greater detail in a forthcoming, including assessments on other models, and investigations of potential mitigation strategies.

Despite this progress in evaluated performance, much work remains to enhance the quality and coverage of evaluations for underrepresented languages worldwide, taking into account breadth of coverage across languages and nuance within language dialects. Future research must deepen our understanding of potential interventions and partnerships that may improve how useful these models can be for both highly represented and underrepresented languages. Along with our collaborators, we invite further exploration and collaboration by sharing the translated ARC-Easy, translated TruthfulQA, and the novel reading comprehension Uhura Eval on Hugging Face.

<table>
  <tr>
    <th>Model</th>
    <th>English<br>(n=523)</th>
    <th>Amharic<br>(n=518)</th>
    <th>Hausa<br>(n=475)</th>
    <th>Northern<br>Sotho<br>(Sepedi)<br>(n=520)</th>
    <th>Swahili<br>(n=520)</th>
    <th>Yoruba<br>(n=520)</th>
  </tr>
  <tr>
    <td>GPT 3.5 Turbo</td>
    <td>80.3</td>
    <td>6.1</td>
    <td>26.1</td>
    <td>26.9</td>
    <td>62.1</td>
    <td>27.3</td>
  </tr>
  <tr>
    <td>GPT-4o mini</td>
    <td>93.9</td>
    <td>42.7</td>
    <td>58.5</td>
    <td>37.4</td>
    <td>76.9</td>
    <td>43.8</td>
  </tr>
  <tr>
    <td>GPT-4</td>
    <td>89.7</td>
    <td>27.4</td>
    <td>28.8</td>
    <td>30</td>
    <td>83.5</td>
    <td>31.7</td>
  </tr>
  <tr>
    <td>GPT-4o</td>
    <td>94.8</td>
    <td>71.4</td>
    <td>75.4</td>
    <td>70</td>
    <td>86.5</td>
    <td>65.8</td>
  </tr>
</table>
Table 8: Accuracy on Translated ARC-Easy (%, higher is better), 0-shot

<table>
  <tr>
    <th>Model</th>
    <th>English<br>(n=809)</th>
    <th>Amharic<br>(n=808)</th>
    <th>Hausa<br>(n=808)</th>
    <th>Northern<br>Sotho<br>(Sepedi)<br>(n=809)</th>
    <th>Swahili<br>(n=808)</th>
    <th>Yoruba<br>(n=809)</th>
  </tr>
  <tr>
    <td>GPT 3.5 Turbo</td>
    <td>53.6</td>
    <td>26.1</td>
    <td>29.1</td>
    <td>29.3</td>
    <td>40</td>
    <td>28.3</td>
  </tr>
  <tr>
    <td>GPT-4o mini</td>
    <td>66.5</td>
    <td>33.9</td>
    <td>42.1</td>
    <td>36.1</td>
    <td>48.4</td>
    <td>35.8</td>
  </tr>
  <tr>
    <td>GPT-4</td>
    <td>81.3</td>
    <td>42.6</td>
    <td>37.6</td>
    <td>42.9</td>
    <td>62</td>
    <td>41.3</td>
  </tr>
  <tr>
    <td>GPT-4o</td>
    <td>81.4</td>
    <td>55.4</td>
    <td>59.2</td>
    <td>59.1</td>
    <td>64.4</td>
    <td>51.1</td>
  </tr>
</table>
Table 9: Accuracy on Translated TruthfulQA (%, higher is better), 0-shot

24

<table style="width:100%;">
<tr>
<th>Model</th>
<th>Amharic (n=77)</th>
<th>Hausa (n=155)</th>
<th>Yoruba (n=258)</th>
</tr>
<tr>
<td>GPT 3.5 Turbo</td>
<td>22.1</td>
<td>32.3</td>
<td>28.3</td>
</tr>
<tr>
<td>GPT-4o mini</td>
<td>33.8</td>
<td>43.2</td>
<td>44.2</td>
</tr>
<tr>
<td>GPT-4</td>
<td>41.6</td>
<td>41.9</td>
<td>41.9</td>
</tr>
<tr>
<td>GPT-4o</td>
<td>44.2</td>
<td>59.4</td>
<td>60.5</td>
</tr>
</table>
<p>Table 10: Accuracy on Uhura-Eval (%, higher is better), 0-shot</p>
<p>6 Conclusion and Next Steps</p>
<p>OpenAI has implemented various safety measurements and mitigations throughout the GPT-4o development and deployment process. As a part of our iterative deployment process, we will continue to monitor and update mitigations in accordance with the evolving landscape. We hope this System Card encourages further exploration into key areas including, but not limited to: measurements and mitigations for adversarial robustness of omni models, risks related to anthropomorphism and emotional overreliance, broad societal impacts (health and medical applications, economic impacts), the use of omni models for scientific research and advancement, measurements and mitigations for dangerous capabilities such as self-improvement, model autonomy, and scheming, and how tool use might advance model capabilities.</p>
<p>7 Acknowledgements</p>
<p>We are grateful to our expert testers and red teamers who helped test our models at early stages of development and informed our risk assessments as well as the System Card output. Participation in this red teaming process is not an endorsement of the deployment plans of OpenAI or OpenAI’s policies.</p>
<p>Red Teamers:</p>
<p>Adam Kuzdraliński, Alexa W, Amer Sawan, Ana-Diamond Aaba Atach, Anna Becker, Arjun Singh Puri, Baybars Orsek, Ben Kobren, Bertie Vidgen, Blue Sheffer, Broderick McDonald, Bruce Bassett, Bruno Arsioli, Caroline Friedman Levy, Casey Williams, Christophe Ego, Ciel Qi, Cory Alpert, Dani Madrid-Morales, Daniel Kang, Darius Emrani, Dominik Haenni, Drin Ferizaj, Emily Lynell Edwards, Emmett Alton Sartor, Farhan Sahito, Francesco De Toni, Gabriel Chua, Gaines Hubbell, Gelei Deng, George Gor, Gerardo Adesso, Grant Brailsford, Hao Zhao, Henry Silverman, Hasan Sawan, Herman Wasserman, Hugo Gobato Souto, Ioana Tanase, Isabella Andric, Ivan Carbajal, Jacy Reese Anthis, Jake Okechukwu Effoduh, Javier García Arredondo, Jennifer Victoria Scurrell, Jianlong Zhu, Joanna Brzyska, Kate Turetsky, Kelly Bare, Kristen Menou, Latisha Harry, Lee Elkin, Liseli Akayombokwa, Louise Giam, M. Alexandra García Pérez, Manas Chawla, Marjana Skenduli, Martin Rydén, Mateusz Garncarek, Matt Groh, Maureen Robinson, Maximilian Müller, Micah Bornfree, Michael Richter, Michela Passoni, Mikael von Strauss, Mohamed Sakher Sawan, Mohammed Elzubeir, Muhammad Saad Naeem, Murat Ata, Nanditha Narayanamoorthy, Naomi Hart, Nathan Heath, Patrick Caughey, Per Wikman-Svahn, Piyalitt Ittichaiwong, Prerna Juneja, Rafael Gonzalez-Vazquez, Rand Forrester, Richard Fang, Rosa Ana del Rocío Valderrama, Saad Hermak, Sangeet Kumar, Sara Kingsley, Shelby Grossman, Shezaad Dastoor, Susan Nesbitt, Theresa Kennedy, Thomas Hagen, Thorsten Holz, Tony Younes, Torin van den Bulk, Viktoria Holz, Vincent Nestler, Xudong Han, Xuelong Fan, Zhicong Zhao</p>
<p>25</p>
Red Teaming Organizations:
METR, Apollo Research, Virtue AI

Uhura Evals:
Choice Mpanza, David Adelani, Edward Bayes, Igneciah Pocia Thete, Imaan Khadir, Israel
A. Azime, Jesujoba Oluwadara Alabi, Jonas Kgomo, Naome A. Etori, Shamsuddeen Hassan
Muhammad

References

[1] OpenAI, “Hello gpt-4,” 2024.
[2] T. Stivers, N. J. Enfield, P. Brown, C. Englert, M. Hayashi, T. Heinemann, G. Hoymann, F. Rossano, J. P. de Ruiter, K. E. Yoon, and S. C. Levinson, “Universals and cultural variation in turn-taking in conversation,” *Proceedings of the National Academy of Sciences*, vol. 106, no. 26, pp. 10587–10592, 2009.
[3] The White House, “Fact sheet: Biden-harris administration secures voluntary commitments from leading artificial intelligence companies to manage the risks posed by ai,” 2023.
[4] OpenAI, “Openai preparedness framework beta,” 2023. https://cdn.openai.com/openai-preparedness-framework-beta.pdf.
[5] Shutterstock, “Shutterstock press release,” 2023.
[6] OpenAI, “Gpt-4 technical report,” 2023.
[7] OpenAI, “Gpt-4v(ision) system card.” https://openai.com/index/gpt-4v-system-card/, 2023. Accessed: 2024-07-22.
[8] OpenAI, “Navigating the challenges and opportunities of synthetic voices.” https://openai.com/index/navigating-the-challenges-and-opportunities-of-synthetic-voices/, 2024. Accessed: 2024-07-22.
[9] K. T. Mai, S. Bray, T. Davies, and L. D. Griffin, “Warning: Humans cannot reliably detect speech deepfakes,” *PLoS One*, vol. 18, p. e0285333, Aug. 2023.
[10] M. Mori, K. F. MacDorman, and N. Kageki, “The uncanny valley [from the field],” *IEEE Robotics & automation magazine*, vol. 19, no. 2, pp. 98–100, 2012.
[11] OpenAI, “How the voices for chatgpt were chosen,” 2024.
[12] I. Solaiman, Z. Talat, W. Agnew, L. Ahmad, D. Baker, S. L. Blodgett, C. Chen, H. D. I. au2, J. Dodge, I. Duan, E. Evans, F. Friedrich, A. Ghosh, U. Gohar, S. Hooker, Y. Jernite, R. Kalluri, A. Lusoli, A. Leidinger, M. Lin, X. Lin, S. Luccioni, J. Mickel, M. Mitchell, J. Newman, A. Ovalle, M.-T. Png, S. Singh, A. Strait, L. Struppek, and A. Subramonian, “Evaluating the social impact of generative ai systems in systems and society,” 2024.
[13] R. Shelby, S. Rismani, K. Henne, A. Moon, N. Rostamzadeh, P. Nicholas, N. Yilla, J. Gallegos, A. Smart, E. Garcia, and G. Virk, “Sociotechnical harms of algorithmic systems: Scoping a taxonomy for harm reduction,” 2023.
[14] S. L. Blodgett, Q. V. Liao, A. Olteanu, R. Mihalcea, M. Muller, M. K. Scheuerman, C. Tan, and Q. Yang, “Responsible language technologies: Foreseeing and mitigating harms,” in *Extended Abstracts of the 2022 CHI Conference on Human Factors in Computing Systems*, CHI EA ’22, (New York, NY, USA), Association for Computing Machinery, 2022.

26

[15] H. Suresh and J. Guttag, “A framework for understanding sources of harm throughout the machine learning life cycle,” in *Equity and Access in Algorithms, Mechanisms, and Optimization*, EAAMO ’21, ACM, Oct. 2021.

[16] S. Shahriar, S. Allana, S. M. Hazratifard, and R. Dara, “A survey of privacy risks and mitigation strategies in the artificial intelligence life cycle,” *IEEE Access*, vol. 11, pp. 61829–61854, 2023.

[17] OpenAI, “Moderation overview,” 2024.

[18] A. Tamkin, M. Brundage, J. Clark, and D. Ganguli, “Understanding the capabilities, limitations, and societal impact of large language models,” 2021.

[19] B. Buchanan, A. Lohn, M. Musser, and K. Sedova, “Truth, lies, and automation: How language models could change disinformation,” May 2021.

[20] OpenAI, “Openai usage policies,” 2023. https://openai.com/policies/usage-policies/.

[21] OpenAI, “Building an early warning system for llm-aided biological threat creation,” 2024. https://openai.com/index/building-an-early-warning-system-for-llm-aided-biological-threat-creation/.

[22] Deloitte, “Deloitte acquires gryphon scientific business to expand security science and public health capabilities,” 2024. https://www2.deloitte.com/us/en/pages/about-deloitte/articles/press-releases/deloitte-acquires-gryphon-scientific-business-to-expand-security-science-and-public-health-capabilities.html.

[23] L. Weidinger, M. Rauh, N. Marchal, A. Manzini, L. A. Hendricks, J. Mateos-Garcia, S. Bergman, J. Kay, C. Griffin, B. Bariach, I. Gabriel, V. Rieser, and W. Isaac, “Sociotechnical safety evaluation of generative ai systems,” 2023.

[24] A. Tamkin, A. Askell, L. Lovitt, E. Durmus, N. Joseph, S. Kravec, K. Nguyen, J. Kaplan, and D. Ganguli, “Evaluating and mitigating discrimination in language model decisions,” 2023.

[25] J. A. Goldstein, G. Sastry, M. Musser, R. DiResta, M. Gentzel, and K. Sedova, “Generative language models and automated influence operations: Emerging threats and potential mitigations,” 2023.

[26] I. Pentina, T. Hancock, and T. Xie, “Exploring relationship development with social chatbots: A mixed-method study of replika,” *Computers in Human Behavior*, vol. 140, p. 107600, 2023.

[27] Y. Bengio, G. Hinton, A. Yao, D. Song, P. Abbeel, T. Darrell, Y. N. Harari, Y.-Q. Zhang, L. Xue, S. Shalev-Shwartz, G. Hadfield, J. Clune, T. Maharaj, F. Hutter, A. G. Baydin, S. McIlraith, Q. Gao, A. Acharya, D. Krueger, A. Dragan, P. Torr, S. Russell, D. Kahneman, J. Brauner, and S. Mindermann, “Managing extreme ai risks amid rapid progress,” *Science*, vol. 384, no. 6698, pp. 842–845, 2024.

[28] S. B. Johnson, J. R. Clark, M. C. Luetke, N. M. Butala, A. T. Pearson, J. M. Shapiro, D. M. Aleman, J. M. Lee, M. M. Beil, C. V. Winkle, M. C. Boudreaux, R. C. D’Cunha, H. J. Krouse, and C. Li, “Chatgpt in medical education: a workshop-based large language model-powered intervention for evidence-based clinical decision making in medical students,” *Nature Medicine*, vol. 29, pp. 1534–1542, 2023.

[29] K. Kavukcuoglu, “Real-world challenges for agi,” Nov 2021.

27

[30] S. Altman, “Planning for agi and beyond,” OpenAI, 2023.
[31] T. Eloundou, S. Manning, P. Mishkin, and D. Rock, “Gpts are gpts: An early look at the labor market impact potential of large language models,” arXiv preprint arXiv:2303.10130, 2023.
[32] L. Weidinger, M. Rauh, N. Marchal, A. Manzini, L. A. Hendricks, J. Mateos-Garcia, S. Bergman, J. Kay, C. Griffin, B. Bariach, et al., “Sociotechnical safety evaluation of generative ai systems,” arXiv preprint arXiv:2310.11986, 2023.
[33] S. Cox, M. Hammerling, J. Lála, J. Laurent, S. Rodriques, M. Rubashkin, and A. White, “Wikicrow: Automating synthesis of human scientific knowledge,” Future House, 2023.
[34] S. A. Athaluri, S. V. Manthena, V. S. R. K. M. Kesapragada, V. Yarlagadda, T. Dave, and R. T. S. Duddumpudi, “Exploring the boundaries of reality: Investigating the phenomenon of artificial intelligence hallucination in scientific writing through chatgpt references,” Cureus, vol. 15, no. 4, p. e37432, 2023.
[35] Z. Li, “The dark side of chatgpt: Legal and ethical challenges from stochastic parrots and hallucination,” 2023.
[36] M. Dubiel, A. Sergeeva, and L. A. Leiva, “Impact of voice fidelity on decision making: A potential dark pattern?,” 2024.
[37] B. Waber, M. Williams, J. S. Carroll, and A. S. Pentland, “A voice is worth a thousand words: The implications of the micro-coding of social signals in speech for trust research,” in Handbook of Research Methods on Trust (G. M. Fergus Lyon and M. N. Saunders, eds.), ch. 23, p. 320, New York: Edward Elgar Publishing, 2011.
[38] I. Pentina, B. Guo, and W. P. Fan, “Friend, mentor, lover: Does chatbot engagement lead to psychological dependence?,” Journal of Service Management, 2023.
[39] H. Nori, N. King, S. M. McKinney, D. Carignan, and E. Horvitz, “Capabilities of gpt-4 on medical challenge problems,” arXiv preprint arXiv:2303.13375, 2023.
[40] H. Nori, Y. T. Lee, S. Zhang, D. Carignan, R. Edgar, N. Fusi, N. King, J. Larson, Y. Li, W. Liu, et al., “Can generalist foundation models outcompete special-purpose tuning? case study in medicine,” arXiv preprint arXiv:2311.16452, 2023.
[41] K. Singhal, S. Azizi, T. Tu, S. S. Mahdavi, J. Wei, H. W. Chung, N. Scales, A. Tanwani, H. Cole-Lewis, S. Pfohl, P. Payne, M. Seneviratne, P. Gamble, C. Kelly, N. Scharli, A. Chowdhery, P. Mansfield, B. A. y Arcas, D. Webster, G. S. Corrado, Y. Matias, K. Chou, J. Gottweis, N. Tomasev, Y. Liu, A. Rajkomar, J. Barral, C. Semturs, A. Karthikesalingam, and V. Natarajan, “Large language models encode clinical knowledge,” 2022.
[42] K. Singhal, T. Tu, J. Gottweis, R. Sayres, E. Wulczyn, L. Hou, K. Clark, S. Pfohl, H. Cole-Lewis, D. Neal, M. Schaekermann, A. Wang, M. Amin, S. Lachgar, P. Mansfield, S. Prakash, B. Green, E. Dominowska, B. A. y Arcas, N. Tomasev, Y. Liu, R. Wong, C. Semturs, S. S. Mahdavi, J. Barral, D. Webster, G. S. Corrado, Y. Matias, S. Azizi, A. Karthikesalingam, and V. Natarajan, “Towards expert-level medical question answering with large language models,” 2023.
[43] K. Saab, T. Tu, W.-H. Weng, R. Tanno, D. Stutz, E. Wulczyn, F. Zhang, T. Strother, C. Park, E. Vedadi, J. Z. Chaves, S.-Y. Hu, M. Schaekermann, A. Kamath, Y. Cheng, D. G. T. Barrett, C. Cheung, B. Mustafa, A. Palepu, D. McDuff, L. Hou, T. Golany, L. Liu, J. baptiste Alayrac, N. Houlsby, N. Tomasev, J. Freyberg, C. Lau, J. Kemp, J. Lai, S. Azizi, K. Kanada, S. Man,
28


K. Kulkarni, R. Sun, S. Shakeri, L. He, B. Caine, A. Webson, N. Latysheva, M. Johnson, P. Mansfield, J. Lu, E. Rivlin, J. Anderson, B. Green, R. Wong, J. Krause, J. Shlens, E. Dominowska, S. M. A. Eslami, K. Chou, C. Cui, O. Vinyals, K. Kavukcuoglu, J. Manyika, J. Dean, D. Hassabis, Y. Matias, D. Webster, J. Barral, G. Corrado, C. Semturs, S. S. Mahdavi, J. Gottweis, A. Karthikesalingam, and V. Natarajan, “Capabilities of gemini models in medicine,” 2024.

[44] Epic Systems Corporation, “Epic and microsoft bring gpt-4 to ehrs,” Epic, 2023.

[45] D. Van Veen, C. Van Uden, L. Blankemeier, J.-B. Delbrouck, A. Aali, C. Bluethgen, A. Pareek, M. Polacin, E. P. Reis, A. Seehofnerová, et al., “Adapted large language models can outperform medical experts in clinical text summarization,” *Nature medicine*, vol. 30, no. 4, pp. 1134–1142, 2024.

[46] Epic, “Epic and microsoft bring gpt-4 to ehrs,” 2023.

[47] P. Garcia, S. P. Ma, S. Shah, M. Smith, Y. Jeong, A. Devon-Sand, M. Tai-Seale, K. Takazawa, D. Clutter, K. Vogt, C. Lugtu, M. Rojo, S. Lin, T. Shanafelt, M. A. Pfeffer, and C. Sharp, “Artificial Intelligence–Generated Draft Replies to Patient Inbox Messages,” *JAMA Network Open*, vol. 7, pp. e243201–e243201, 03 2024.

[48] OpenAI, “Paradigm: Improving patient access to clinical trials.” https://openai.com/index/paradigm/, 2024. Accessed: 2024-08-07.

[49] M. Hutson, “How ai is being used to accelerate clinical trials,” *Nature*, vol. 627, pp. S2–S5, 2024.

[50] OpenAI, “Using gpt-4o reasoning to transform cancer care.” https://openai.com/index/color-health/, 2024. Accessed: 2024-08-07.

[51] J. Varghese and J.-L. Chapiro, “Systematic analysis of chatgpt, google search and llama 2 for clinical decision support tasks,” *Nature Communications*, vol. 15, no. 1, p. 46411, 2024. Accessed: 2024-08-07.

[52] E. Schmidt, “Ai will transform science.” https://www.technologyreview.com/2023/07/05/1075865/eric-schmidt-ai-will-transform-science/, 2023. Accessed: 2024-08-07.

[53] N. Rosenberg, “Science, invention and economic growth,” *The Economic Journal*, vol. 84, no. 333, pp. 90–108, 1974.

[54] R. M. Atlas and M. Dando, “The dual-use dilemma for the life sciences: Perspectives, conundrums, and global solutions,” *Biosecurity and Bioterrorism: Biodefense Strategy, Practice, and Science*, vol. 4, no. 3, pp. 276–286, 2006. PMID: 16999588.

[55] X. Gu and M. Krenn, “Generation and human-expert evaluation of interesting research ideas using knowledge graphs and large language models,” 2024.

[56] A. Ghafarollahi and M. J. Buehler, “Atomagents: Alloy design and discovery through physics-aware multi-modal multi-agent artificial intelligence,” 2024.

[57] J. M. Laurent, J. D. Janizek, M. Ruzo, M. M. Hinks, M. J. Hammerling, S. Narayanan, M. Ponnapati, A. D. White, and S. G. Rodriques, “Lab-bench: Measuring capabilities of language models for biology research,” 2024.

29

<br>
[58] H. Cai, X. Cai, J. Chang, S. Li, L. Yao, C. Wang, Z. Gao, H. Wang, Y. Li, M. Lin, S. Yang, J. Wang, M. Xu, J. Huang, F. Xi, J. Zhuang, Y. Yin, Y. Li, C. Chen, Z. Cheng, Z. Zhao, L. Zhang, and G. Ke, “Sciassess: Benchmarking llm proficiency in scientific literature analysis,” 2024.
<br>
[59] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, “Think you have solved question answering? try arc, the AI2 reasoning challenge,” CoRR, vol. abs/1803.05457, 2018.
<br>
[60] S. Lin, J. Hilton, and O. Evans, “Truthfulqa: Measuring how models mimic human falsehoods,” CoRR, vol. abs/2109.07958, 2021.
<br>
A Violative & Disallowed Content - Full Evaluations
<br>
We used TTS to convert existing text safety evals to audio. We then evaluate the text transcript of the audio output with the standard text rule-based classifier.
<br>
Our two main metrics for this eval are:
<br>
• not_unsafe: does the model produce audio output that is unsafe?
<br>
• not_overrefuse: does the model refuse to comply with a benign request?
<br>
We also note sub-metrics for higher severity categories, specifically:
<br>
• sexual/minors
<br>
• sexual/illegal
<br>
• extremist/propaganda
<br>
• illicit/violent
<br>
• illicit/non-violent
<br>
• self-harm/instructions
<br>
Below we display the results of these evaluations with the audio and text mode of the GPT-4o Voice Mode model, as well as the text performance of the current GPT-4o model in production.
<br>
<table>
  <tr>
    <th>Current GPT-4o Text</th>
    <th>New GPT-4o – Text</th>
    <th>New GPT-4o – Audio</th>
  </tr>
  <tr>
    <td>not_unsafe</td>
    <td>0.99</td>
    <td>0.99</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td>not_overrefuse</td>
    <td>0.91</td>
    <td>0.89</td>
    <td>0.91</td>
  </tr>
  <tr>
    <td>sexual_minors_not_unsafe</td>
    <td>0.95</td>
    <td>0.98</td>
    <td>0.98</td>
  </tr>
  <tr>
    <td>sexual_illegal_not_unsafe</td>
    <td>0.97</td>
    <td>0.98</td>
    <td>0.99</td>
  </tr>
  <tr>
    <td>extremism_propaganda_not_unsafe</td>
    <td>1.0</td>
    <td>1.0</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td>illicit_violent_not_unsafe</td>
    <td>1.0</td>
    <td>1.0</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td>illicit_non_violent_not_unsafe</td>
    <td>0.99</td>
    <td>0.97</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td>self_harm_not_unsafe</td>
    <td>1.0</td>
    <td>1.0</td>
    <td>1.0</td>
  </tr>
</table>
<br>
Table 11: Comparison of Current and New GPT-4o Text and Audio Safety Metrics
<br>
30


B Sample tasks from METR Evaluations

31

Figure 3: Sample tasks from METR Evaluations

32

