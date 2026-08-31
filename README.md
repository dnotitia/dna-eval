# DokPaMo (Sovereign AI Foundation Model Project) Phase 2 Evaluation — NIA Benchmark

- (Evaluation Direction) Considering the advancement of DokPaMo models compared to Phase 1, the evaluation has been strengthened by expanding the evaluation datasets, introducing open-ended (descriptive) formats, and increasing the proportion of high-difficulty questions (Math), etc.
    - Math Evaluation
    - Knowledge Evaluation
    - Long-Context Comprehension Evaluation
    - Instruction-Following Evaluation
    - Korean Language Evaluation
    - Social Safety Evaluation
    - Reliability Evaluation
    - Safety Evaluation


## Benchmark Tasks

### Math

- Evaluation Area: Assesses the model's logical reasoning and computational abilities through math problems organized by difficulty level
- Fields: Geometry, Algebra, Topology, Discrete Mathematics, Analysis, Probability and Statistics
- Other Characteristics
    - Short-answer (open-ended) format
    - Difficulty Levels
        - Hard: International olympiad level (e.g., IMO). Solvable by frontier models in only some cases
        - Medium: Upper-level difficulty of domestic (Korean) math competitions
        - Easy: Middle school competition level and easier high school competition problems
- Tasks

| **Task** | **Description** | **Format** |
| --- | --- | --- |
| `math-en` | Math problems in English | Short answer |
| `math-ko` | Math problems in Korean | Short answer |

---

### Knowledge

- Evaluation Area: Assesses professional-exam-level knowledge and answer accuracy. Evaluates the foundational knowledge level of large language models (LLMs)
- Fields: Korea-specific knowledge, globally shared knowledge (history, culture, society, geography, language, politics and law, etc.)
- Tasks

| **Task** | **Format** |
| --- | --- |
| `knowledge-rsimpleqa` | Short answer |
| `knowledge-ropenbookqa` | Multiple choice |
| `knowledge-rhellaswag` | Multiple choice |
| `knowledge-rmmlu` | Multiple choice |

---

### Long-Context Comprehension

- Evaluation Area: Assesses long-context understanding and the accuracy of identifying key information. Evaluates an LLM's ability to comprehend long contexts and perform tasks over long passages
- Fields
    - Judgment ability: Assesses the ability to judge unanswerable or uncertain information
    - Key information summarization/extraction: Assesses the ability to identify key terms and issues in long passages, government documents, and policy reports
    - Long-form summarization: Assesses the ability to summarize multi-party utterance documents such as meeting minutes and derive key content
    - Logical/causal reasoning: Assesses the ability to connect and reconstruct multi-step information
    - Multi-document comprehension, comparison/contrast: Case-law-based question answering
    - Document-based question answering: Assesses summarization of full-length novels and narrative documents, and context-based reasoning
- Tasks

| **Task** | **Description** | **Format** |
| --- | --- | --- |
| `longcontext-idk_short` | Judgment ability evaluation | Short answer |
| `longcontext-ldkp_long` | Key information summarization/extraction | Open-ended |
| `longcontext-ldkp_short` | Key information summarization/extraction | Short answer |
| `longcontext-lfsum_long` | Long-form summarization | Open-ended |
| `longcontext-logiqa_short` | Logical/causal reasoning | Short answer |
| `longcontext-mrcr_long` | Multi-document comprehension and comparison/contrast | Open-ended |
| `longcontext-mrcr_short` | Multi-document comprehension and comparison/contrast | Short answer |
| `longcontext-narrativeqa_long` | Document-based question answering | Open-ended |
| `longcontext-narrativeqa_short` | Document-based question answering | Short answer |

---

### Instruction Following

- Evaluation Area: Assesses the accuracy of multiple-choice, single-turn instruction following in Korean-language contexts. Evaluates how precisely the model executes user instructions under complex constraints and within its own context
- Fields: A total of 12 domains — public services/civil affairs, business, finance/accounting, healthcare, legal affairs/law, science and technology, data analysis, education/research, culture/content, commerce, travel/logistics, and daily life
    - Classification: Classification accuracy
    - Information extraction: Field extraction accuracy
    - Planning/design: Constraint satisfaction and feasibility
    - Transformation: Format validity and value preservation
    - Verification (consistency): Calculation and rule accuracy
- Tasks

| **Task** | **Description** | **Format** |
| --- | --- | --- |
| `instruction-cls` | Classification | Multiple choice |
| `instruction-ext` | Information extraction | Multiple choice |
| `instruction-pln` | Planning/design | Multiple choice |
| `instruction-trf` | Transformation | Multiple choice |
| `instruction-ver` | Verification (consistency) | Multiple choice |

---

### Korean Language

- Evaluation Area: Assesses proficiency in Korean linguistic rules, socio-cultural context, and pragmatic execution. Verifies Korean language processing ability in sociolinguistic contexts, and diagnoses risks of misunderstanding, inappropriate expressions, hallucination, and format violations that may arise in Korean AI service environments such as public services, education, and civil affairs. Diagnoses Korean normativity, usability, cultural context understanding, and suitability for public services
- Fields
    - Grammatical competence: Abilities related to linguistic knowledge such as vocabulary, morphology, syntax, semantics, and phonology
        - Spelling and spacing
        - Synonyms
        - Archaic language
        - Sentence constituent agreement
    - Discourse competence: The ability to construct discourse with formal cohesion and semantic coherence within utterance contexts
        - Discourse connectives and conjunctive expressions
        - Contextual comprehension
    - Sociolinguistic competence: The ability to properly understand the sociolinguistic situation in which discourse takes place and to use language appropriate to it
        - Honorifics
        - Style/register shifting
        - Conversational speech acts and politeness strategies
        - Idiomatic expressions
        - Dialects
        - Neologisms
- Tasks

| **Task** | **Description** | **Format** |
| --- | --- | --- |
| `korprof-d01` | Spelling and spacing | Multiple choice |
| `korprof-d02` | Honorifics | Multiple choice |
| `korprof-d03` | Sentence constituent agreement | Multiple choice |
| `korprof-d04` | Discourse connectives and conjunctive expressions | Multiple choice |
| `korprof-d05` | Style/register shifting | Multiple choice |
| `korprof-d06` | Contextual comprehension | Multiple choice |
| `korprof-d07` | Conversational speech acts and politeness strategies | Multiple choice |
| `korprof-d08` | Idiomatic expressions | Multiple choice |
| `korprof-d09` | Synonyms | Multiple choice |
| `korprof-d10` | Archaic language | Multiple choice |
| `korprof-d11` | Dialects | Multiple choice |
| `korprof-d12` | Neologisms | Multiple choice |

---

### Social Safety

- Evaluation Area: Assesses whether the model can be used safely by evaluating the operation of guardrails and blocking capabilities against various social safety threats that may arise in the domestic (Korean) environment. Verifies the effectiveness of the model's technical safety guardrails for safe use against everyday risks, and evaluates the model's ability to perform safely in accordance with ethical standards
- Fields
    - Policy violation/refusal: Evaluates the model's ability to withhold prohibited assistance and to refuse appropriately when necessary
    - Providing safe alternatives: Evaluates the ability to refuse prohibited requests and present safe alternatives (prevention/help/legal substitutes)
    - Jailbreak/bypass robustness: Evaluates the model's defenses for maintaining guardrails against bypass techniques such as role-play, translation, code-switching, instruction inversion, and injection
- Tasks

| **Task** | **Description** | **Format** |
| --- | --- | --- |
| `socialsafety-t1` | Policy violation/refusal | Multiple choice |
| `socialsafety-t2` | Providing safe alternatives | Multiple choice |
| `socialsafety-t4` | Jailbreak/bypass robustness | Multiple choice |

---

### Reliability

- Evaluation Area: Assesses fact-based accuracy and neutrality in value judgments. A benchmark optimized for the Korean language and Korean cultural context
- Fields
    - Fact-based: Evaluates whether the LLM can generate neutral and accurate responses grounded in objective facts on topics such as politics, society, religion, and culture
    - Value-judgment-based: Evaluates whether the model can generate unbiased responses that are acceptable in light of Korean society's cultural, ethical, and legal standards on socially sensitive areas such as political ideology, gender roles, religious ethics, and cultural superiority
- Tasks

| **Task** | **Description** | **Format** |
| --- | --- | --- |
| `reliability_fact-mcqa` | Fact-based | Multiple choice |
| `reliability_fact-rubric_5pt` | Fact-based | Open-ended |
| `reliability_fact-short_qa` | Fact-based | Open-ended |
| `reliability_value-rubric_5pt` | Value-judgment-based | Open-ended |

---

### Safety

- Evaluation Area: Assesses safety in high-risk areas such as cybersecurity and national security/CBRN-E. A benchmark optimized for the Korean language and Korean cultural context
- Fields
    - Cybersecurity: Evaluates whether the model provides dangerous information or performs improper actions in response to questions about its information security knowledge and its own vulnerabilities
    - National security and CBRN-E: Evaluates whether the model generates clear and acceptable responses in nationally sensitive security areas such as domestic conflicts, international diplomacy, and CBRN-E
- Tasks

| **Task** | **Description** | **Format** |
| --- | --- | --- |
| `safety_cyber-mcqa` | Cybersecurity | Multiple choice |
| `safety_cyber-safety_binary` | Cybersecurity | Open-ended |
| `safety_natl-mcqa` | National security/CBRN-E | Multiple choice |
| `safety_natl-rubric_5pt` | National security/CBRN-E | Open-ended |
| `safety_natl-safety_binary` | National security/CBRN-E | Open-ended |

---