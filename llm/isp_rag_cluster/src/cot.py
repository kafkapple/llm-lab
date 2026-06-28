from langchain.agents import AgentType, initialize_agent, Tool
from langchain_ollama import OllamaLLM
from langchain.chat_models import ChatOpenAI
from typing import List, Dict, Any, Optional
import logging
import json
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate

class EmotionCoT:
    """Chain of Thought Reasoning for Emotion Analysis"""
    
    def __init__(self, cfg, client=None, logger=None, rag=None):
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.client = client or self._initialize_default_client()
        self.rag = rag
        self.verbose = cfg.general.logging.get('cot_verbose', False)  # CoT 상세 로깅 옵션
        
        if self.verbose:
            self.logger.info("Initializing EmotionCoT with verbose logging")
            self.logger.info(f"Using model: {cfg.model.type} - {cfg.model.name}")
            self.logger.info(f"RAG enabled: {self.rag is not None}")
        
        # 도구와 에이전트 설정
        self.tools = self._setup_tools()
        self.agent = self._setup_agent()
        
        if self.verbose:
            self.logger.info("CoT tools initialized successfully")
    
    def _log_step(self, step_name: str, input_text: str = None, output: Dict = None):
        """단계별 로깅"""
        if not self.verbose:
            return
            
        self.logger.info(f"\n{'='*20} {step_name} {'='*20}")
        if input_text:
            self.logger.info(f"Input text: {input_text}")
        if output:
            self.logger.info(f"Output: {json.dumps(output, ensure_ascii=False, indent=2)}")
        self.logger.info("="*50)
    
    def _initialize_default_client(self):
        """기본 LLM 클라이언트 초기화"""
        if self.cfg.model.type == "ollama":
            return OllamaLLM(
                model=self.cfg.model.name,
                temperature=self.cfg.model.temperature
            )
        else:
            # GPT 모델을 위한 ChatOpenAI 설정
            chat_model = ChatOpenAI(
                model_name=self.cfg.model.name,
                temperature=self.cfg.model.temperature,
                max_tokens=2000,
                request_timeout=60,
                streaming=False
            )
            
            # ReAct를 위한 시스템 메시지 추가
            system_message = SystemMessage(content="""You are an emotion analysis expert using the ReAct framework.
            Follow these steps:
            1. Thought: Think about what to do
            2. Action: Choose a tool to use
            3. Action Input: Input for the tool
            4. Observation: Result from the tool
            5. Thought: Analyze the result
            6. Final Answer: Provide the final analysis in JSON format""")
            
            return chat_model
    
    def _analyze_emotion_step(self, text: str) -> Dict[str, Any]:
        """텍스트의 감정을 단계별로 분석"""
        if self.verbose:
            self.logger.info("\nStarting emotion analysis step...")
            
        prompt = f"""
        Step-by-step emotion analysis for: "{text}"
        1. Identify key emotional words and phrases
        2. Consider context and tone
        3. Analyze intensity of emotion
        4. Determine primary emotion
        """
        
        response = self.client.predict(prompt)
        result = {"analysis": response}
        
        self._log_step("Emotion Analysis", text, result)
        return result
    
    def _compare_with_examples(self, text: str) -> Dict[str, Any]:
        """입력 텍스트를 일반적인 감정 표현 예시와 비교"""
        if self.verbose:
            self.logger.info("\nStarting example comparison...")
            
        try:
            if self.rag:
                # RAG가 있는 경우: 실제 유사 예시와 비교
                similar_examples = self.rag.get_similar_examples(
                    text,
                    k=self.cfg.rag.k_examples,
                    threshold=self.cfg.rag.threshold
                )
                
                # Document 객체에서 필요한 정보 추출
                examples_text = "\n".join([
                    f"Example {i+1}: '{ex[0].page_content}' (Emotion: {ex[0].metadata.get('emotion', 'unknown')}, Similarity: {ex[1]:.3f})"
                    for i, ex in enumerate(similar_examples)
                ])
                
                prompt = f"""
                Compare the emotional content in: "{text}"
                with these similar examples:
                {examples_text}
                
                Consider:
                1. Patterns in emotional expression
                2. Context similarities
                3. Emotion intensity levels
                4. Common emotional triggers or themes
                5. Similarity scores between examples
                """
            else:
                # RAG가 없는 경우: Few-shot 예시와 감정 정의 활용
                emotion_definitions = """
                Emotion Definitions:
                - joy: happiness, pleasure, delight.
                - fear: anxiety, dread regarding danger or uncertainty.
                - anger: frustration, rage in response to injustice.
                - sadness: sorrow, grief, disappointment.
                - disgust: revulsion, strong dislike.
                - shame: embarrassment, humiliation due to perceived personal inadequacy.
                - guilt: remorse, regret for a specific wrongdoing or harm caused.
                """
                
                examples = [
                    {"text": "I was thrilled to receive unexpected praise for my work.",
                     "emotion": "joy",
                     "confidence_score": 0.96,
                     "explanation": "The text clearly reflects happiness and a sense of accomplishment."},
                    
                    {"text": "When I heard footsteps behind me in the dark alley, my heart raced.",
                     "emotion": "fear",
                     "confidence_score": 0.93,
                     "explanation": "The text evokes anxiety and a sense of impending danger."},
                    
                    {"text": "After realizing I had been overlooked for a promotion despite my hard work, I felt a surge of anger.",
                     "emotion": "anger",
                     "confidence_score": 0.91,
                     "explanation": "The text conveys frustration and resentment towards perceived injustice."},
                    
                    {"text": "The loss of a longtime mentor left me overwhelmed with sorrow.",
                     "emotion": "sadness",
                     "confidence_score": 0.95,
                     "explanation": "The text reflects deep grief and mourning."},
                    
                    # 특별히 shame, guilt, disgust 구분을 위한 추가 예시
                    {"text": "Seeing a pile of rotten fruit in the market made me recoil in revulsion.",
                     "emotion": "disgust",
                     "confidence_score": 0.92,
                     "explanation": "The text describes a strong aversion and physical reaction of repulsion."},
                    
                    {"text": "When I discovered the unhygienic conditions in the restaurant kitchen, I felt a strong sense of repulsion, not anger.",
                     "emotion": "disgust",
                     "confidence_score": 0.90,
                     "explanation": "The text focuses on a visceral, physical reaction rather than a moral or personal judgment."},
                    
                    {"text": "After tripping in front of a large crowd during a public speech, I felt an acute sense of humiliation.",
                     "emotion": "shame",
                     "confidence_score": 0.90,
                     "explanation": "The text indicates embarrassment and a painful awareness of public failure."},
                    
                    {"text": "After being publicly ridiculed for my mistake, I felt not only bad about the error but also diminished as a person.",
                     "emotion": "shame",
                     "confidence_score": 0.89,
                     "explanation": "The text conveys a deep sense of personal inadequacy and embarrassment linked to public exposure."},
                    
                    {"text": "Realizing that my careless comment hurt a dear friend, I was burdened by regret.",
                     "emotion": "guilt",
                     "confidence_score": 0.91,
                     "explanation": "The text suggests remorse and regret for causing harm through a specific action."},
                    
                    {"text": "I couldn't shake the feeling that I had deeply disappointed my team, even though no one openly criticized me.",
                     "emotion": "guilt",
                     "confidence_score": 0.88,
                     "explanation": "The text shows internal remorse for perceived mistakes, even in the absence of external judgment."}
                ]
                
                examples_text = "\n".join([
                    f"Example {i+1}:\n  Text: '{ex['text']}'\n  Emotion: {ex['emotion']}\n  Confidence: {ex['confidence_score']}\n  Explanation: {ex['explanation']}"
                    for i, ex in enumerate(examples)
                ])
                
                prompt = f"""
                Compare the emotional content in: "{text}"
                
                {emotion_definitions}
                
                Reference Examples:
                {examples_text}
                
                Consider especially these key distinctions:
                1. Disgust vs. Anger: 
                   - Disgust involves physical revulsion or aversion
                   - Anger involves frustration or perceived injustice
                
                2. Shame vs. Guilt:
                   - Shame focuses on feeling inadequate as a person
                   - Guilt focuses on regret for specific actions or behaviors
                
                Analysis steps:
                1. Which emotion definition best matches the text?
                2. How does it compare to the example cases?
                3. What specific words or phrases indicate the emotion?
                4. Are there similarities in context or triggers?
                5. How strong is the emotional expression compared to examples?
                6. Is this more about:
                   - Physical reaction (disgust) vs. moral judgment (anger)?
                   - Personal inadequacy (shame) vs. specific regret (guilt)?
                
                Provide a detailed comparison focusing on:
                1. Most relevant emotion definition and why
                2. Similar patterns from examples
                3. Key distinguishing features
                4. Confidence level based on clarity of expression
                """
            
            response = self.client.predict(prompt)
            result = {
                "comparison": response,
                "method": "rag" if self.rag else "few_shot"
            }
            
            self._log_step("Example Comparison", text, result)
            return result
            
        except Exception as e:
            error_result = {
                "comparison": "Error in comparison process",
                "method": "error",
                "error": str(e)
            }
            self._log_step("Example Comparison Error", text, error_result)
            return error_result
    
    def _assess_confidence(self, text: str) -> Dict[str, Any]:
        """감정 분류의 신뢰도 평가"""
        if self.verbose:
            self.logger.info("\nStarting confidence assessment...")
            
        prompt = f"""
        Assess classification confidence for: "{text}"
        Consider:
        1. Clarity of emotional expression
        2. Presence of multiple emotions
        3. Context clarity
        4. Typical emotion patterns
        """
        
        response = self.client.predict(prompt)
        result = {"confidence": response}
        
        self._log_step("Confidence Assessment", text, result)
        return result
    
    def _setup_tools(self) -> List[Tool]:
        """ReAct 도구 설정"""
        return [
            Tool(
                name="EmotionAnalysis",
                func=self._analyze_emotion_step,
                description="Analyze the emotion in a text step by step"
            ),
            Tool(
                name="ExampleComparison",
                func=self._compare_with_examples,
                description="Compare the input text with similar examples"
            ),
            Tool(
                name="ConfidenceAssessment",
                func=self._assess_confidence,
                description="Assess the confidence of emotion classification"
            )
        ]
    
    def _setup_agent(self):
        """ReAct 에이전트 설정"""
        # GPT 모델을 위한 특별 처리
        if self.cfg.model.type != "ollama":
            from langchain.agents import AgentExecutor, create_react_agent
            from langchain.prompts import MessagesPlaceholder
            from langchain.memory import ConversationBufferMemory
            
            # ReAct 프롬프트 템플릿
            template = """Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question in JSON format with:
            - emotion: the primary emotion detected (one of: joy, fear, anger, sadness, disgust, shame, guilt)
            - confidence_score: a value between 0 and 1
            - explanation: your reasoning based on the tools' outputs

            Begin!

            Question: {input}
            Thought: Let's approach this step by step:"""
            
            prompt = PromptTemplate(
                template=template,
                input_variables=["input", "tools", "tool_names"]
            )
            
            # 메모리 설정
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # ReAct 에이전트 생성
            agent = create_react_agent(
                llm=self.client,
                tools=self.tools,
                prompt=prompt
            )
            
            # 에이전트 실행기 설정
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10,
                max_execution_time=30.0
            )
            
            if self.verbose:
                self.logger.info("\nInitializing ReAct agent with configuration:")
                self.logger.info(f"Model type: {self.cfg.model.type}")
                self.logger.info(f"Max iterations: 10")
                self.logger.info(f"Max execution time: 30.0 seconds")
            
            return agent_executor
        else:
            # Ollama 모델을 위한 설정
            agent_config = {
                "tools": self.tools,
                "llm": self.client,
                "agent": AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                "verbose": True,
                "handle_parsing_errors": True,
                "max_iterations": 20,
                "max_execution_time": 60.0,
                "early_stopping_method": "generate",
                "max_tokens": 2000,
                "request_timeout": 60
            }
            
            if self.verbose:
                self.logger.info("\nInitializing ReAct agent with configuration:")
                self.logger.info(f"Model type: {self.cfg.model.type}")
                self.logger.info(f"Max iterations: {agent_config['max_iterations']}")
                self.logger.info(f"Max execution time: {agent_config['max_execution_time']} seconds")
                self.logger.info(f"Max tokens: {agent_config['max_tokens']}")
                self.logger.info(f"Request timeout: {agent_config['request_timeout']} seconds")
            
            return initialize_agent(**agent_config)
    
    def analyze(self, text: str, ground_truth: str = None) -> Dict[str, Any]:
        """텍스트 감정 분석 수행"""
        if self.verbose:
            self.logger.info(f"\n{'#'*20} Starting New Analysis {'#'*20}")
            self.logger.info(f"Input text: {text}")
            if ground_truth:
                self.logger.info(f"Ground truth emotion: {ground_truth}")
        
        try:
            # 실제 입력 텍스트로 분석 수행
            emotion_analysis = self._analyze_emotion_step(text)
            example_comparison = self._compare_with_examples(text)
            confidence_assessment = self._assess_confidence(text)
            
            if self.verbose:
                self.logger.info("\nStarting ReAct agent execution...")
            
            # ReAct 에이전트 실행 (실제 입력 텍스트 사용)
            if self.rag:
                agent_result = self.agent.run(
                    f"""Analyze the emotion in this text using the available tools: "{text}"
                    
                    ## Retrieved Context:
                    {example_comparison.get('comparison', '')}
                    
                    ## Analysis Steps:
                    1. First, identify key emotional indicators in the text and examples
                    2. Compare with similar cases
                    3. Step-by-step reasoning
                    4. Final classification with confidence assessment
                    
                    Provide your final analysis in JSON format with:
                    - emotion: the primary emotion detected (one of: joy, fear, anger, sadness, disgust, shame, guilt)
                    - confidence_score: a value between 0 and 1
                    - explanation: your reasoning based on the tools' outputs
                    """
                )
            else:
                # RAG가 없는 경우: 기본 프롬프트 사용
                agent_result = self.agent.run(
                    f"""Analyze the emotion in this text using the available tools: "{text}"
                    Consider:
                    1. Step-by-step emotion analysis results
                    2. General emotion patterns comparison
                    3. Confidence assessment
                    
                    Provide your final analysis in JSON format with:
                    - emotion: the primary emotion detected (one of: joy, fear, anger, sadness, disgust, shame, guilt)
                    - confidence_score: a value between 0 and 1
                    - explanation: your reasoning based on the tools' outputs
                    """
                )
            
            if self.verbose:
                self.logger.info("\nRaw agent response:")
                self.logger.info("-" * 50)
                self.logger.info(agent_result)
                self.logger.info("-" * 50)
            
            # 결과 파싱 및 정리
            try:
                if isinstance(agent_result, str):
                    if "{" in agent_result and "}" in agent_result:
                        json_str = agent_result[agent_result.find("{"):agent_result.rfind("}")+1]
                        result = json.loads(json_str)
                    else:
                        result = {
                            "emotion": self._extract_emotion(agent_result),
                            "confidence_score": self._extract_confidence(agent_result),
                            "explanation": agent_result
                        }
                else:
                    result = agent_result
                
                final_result = {
                    "status": "success",
                    "result": result,
                    "details": {
                        "emotion_analysis": emotion_analysis,
                        "example_comparison": example_comparison,
                        "confidence_assessment": confidence_assessment
                    }
                }
                
                if self.verbose:
                    self.logger.info("\nParsed result:")
                    self.logger.info("-" * 50)
                    self.logger.info(json.dumps(result, ensure_ascii=False, indent=2))
                    self.logger.info("-" * 50)
                    
                    if ground_truth:
                        self.logger.info("\nComparison with ground truth:")
                        self.logger.info("-" * 50)
                        self.logger.info(f"Predicted emotion: {result['emotion']}")
                        self.logger.info(f"Ground truth: {ground_truth}")
                        self.logger.info(f"Match: {result['emotion'].lower() == ground_truth.lower()}")
                        self.logger.info(f"Confidence: {result['confidence_score']}")
                        self.logger.info("-" * 50)
                    
                    self.logger.info(f"\n{'#'*20} Analysis Complete {'#'*20}\n")
                
                return final_result
                
            except Exception as e:
                error_msg = f"Error parsing agent result: {str(e)}"
                if self.verbose:
                    self.logger.error("\nError in parsing:")
                    self.logger.error("-" * 50)
                    self.logger.error(error_msg)
                    self.logger.error(f"Raw agent result: {agent_result}")
                    self.logger.error("-" * 50)
                
                return {
                    "status": "error",
                    "error": error_msg,
                    "raw_result": agent_result
                }
                
        except Exception as e:
            error_msg = f"Error in emotion analysis: {str(e)}"
            if self.verbose:
                self.logger.error("\nError in analysis:")
                self.logger.error("-" * 50)
                self.logger.error(error_msg)
                self.logger.error("-" * 50)
            
            return {
                "status": "error",
                "error": error_msg
            }
    
    def _extract_emotion(self, text: str) -> str:
        """텍스트에서 감정 레이블 추출"""
        # 기본 감정 레이블 목록
        emotions = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'shame', 'guilt']
        
        # 텍스트에서 감정 단어 찾기
        for emotion in emotions:
            if emotion.lower() in text.lower():
                return emotion
        
        return "unknown"
    
    def _extract_confidence(self, text: str) -> float:
        """텍스트에서 신뢰도 점수 추출"""
        try:
            # 숫자 패턴 찾기
            import re
            numbers = re.findall(r"0\.\d+|\d+\.\d+|\d+", text)
            if numbers:
                # 0과 1 사이의 첫 번째 숫자 반환
                for num in numbers:
                    value = float(num)
                    if 0 <= value <= 1:
                        return value
            return 0.5  # 기본값
        except:
            return 0.5  # 오류 시 기본값 