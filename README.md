## 목차

---

## NMT with Transformers

(진행 중)

### 프로젝트 개요
- PLM 모델을 활용한 실사용 목적의 번역기를 만들기 위한 기술 요건을 파악한다.

### 개발환경 및 사용기술
- AWS EC2, Ubuntu, Conda, Git/Github, DVC, WandB, 
- PyTorch, Huggingface, Papermill

### 예정사항
- 
- 
- 

---

## Backend API for Chatbot with Poly Encoder

### 프로젝트 개요
- 챗봇 서비스를 위한 DB 연동 REST API를 개발한다

### 개발환경 및 사용기술
- Colab, WSL2, Ubuntu, Conda, Git/Github
- MySQL, SQLAlchemy, Flask, Marshmallow, Scrapy, Selenium

### 수행사항
- Git/Github을 통한 프로젝트 관리를 적용
- 폴리 인코더 챗봇 함수 정리 및 모듈화
- MySQL 서버 구동 및 챗봇 데이터 이전
- SQLAlchemy를 활용한 ORM 코드 작성
- Flask와 Marshmallow를 통한 REST API 구현
- Postman을 사용한 API 시험

### 회고
- 실제 서비스 상황을 염두에 두고 DB를 연동한 REST API를 구현한 프로젝트
- 단순히 백업용으로 사용하던 Git/Github을 프로젝트 관리에 적용하고자 버전 관리 요령, Merge, Rebase, Pull Request, Conflict Resolve 등에 대해 학습함
- 프로젝트를 진행하며 Commit Convention에 대해 인지하게 되었고, Commit Message를 좀 더 세련화하기 위한 고민을 이어나감
- 파이썬으로 REST API를 구현하기 위한 기술인 Django/Django REST framework과 Flask/SQL Alchemy/Marshmallow 중 개념 학습에 더 효과적이라고 느껴졌던 후자를 선택
- Ubuntu에 MySQL을 설치하고 파이썬으로 접근하기 위한 DBAPI를 탐색
- MySQL Connector/Python과 PyMySQL 중 Oracle사의 공식 배포인 전자를 선택
- SQL을 사용해 Database를 생성하고 User 권한을 조정
- PyCon 세미나, API 문서, 공식 튜토리얼을 참고하여 SQLAlchemy에 대한 학습을 진행
- Flask-SQLAlchemy, Flask-Marshmallow를 활용한 REST API 작성 방법을 학습하고, Marshmallow의 Serialization과 Validation 개념을 이해
- Flask Extentions를 사용한 구현에는 생략된 절차가 많았기에 학습 목적에 부합하지 않는다고 판단되어 사용하지 않기로 결정함
- SQLAlchemy와 MySQL Connector/Python를 활용해 챗봇 답변 목록을 DB에 전송하고, Session을 통한 쿼리 작업을 구현
- 모듈화를 진행하며 REST API를 작성하고 Postman을 사용한 시험을 진행
- 모델 추론과 가장 긴밀하게 연결되어 있는 Backend 기술에 대해 알아가며 AI를 실제 서비스화하기 위해 필요한 다양한 업무들을 살펴보고 이해할 수 있었음

### 참고자료

---

## Retrieval-based Chatbot with Poly Encoder

### 프로젝트 개요
- AIHub의 감성대화 데이터로 폴리 인코더를 훈련시키고 리트리벌 챗봇을 구현한다.

### 모델 / 데이터
- klue/bert-base / Poly Encoder
- AIHub 감성대화 Train 40879 / Eval 5130

### 개발환경 및 사용기술
- Colab, AWS EC2, Ubuntu, Conda, Git/Github
- PyTorch, Huggingface, Pandas

### 수행사항
- 데이터 리포맷
- 훈련 데이터 검증
- 적정 Poly-M 설정
- 훈련 및 성능 평가 지표 측정
- 모델 활용 방안 강구
- IR System 구축을 위한 함수 작성
- 후보 임베딩 사전 산출
- 유사도 점수 계산 및 GPU 자원 관리
- 터미널 구동 챗봇 UI 작성

### 회고
- 지난주의 학습을 바탕으로 챗봇 구현을 위해 본 프로젝트를 시작
- 구현을 통해 기획을 검증해야 했기 때문에 끊임없이 도전해야 했던 한 주
- 폴리 인코더 훈련을 위해 한국어 데이터로 사전 학습된 klue/bert-base를 선택
- 훈련에 적합한 형식으로 변환 가능했던 AIHub 감성대화를 리포맷
- 데이터를 최대한 활용하기 위해 정수 인코딩 후의 시퀀스 길이를 살펴보고 그에 맞춰 토크나이저를 재설정
- Poly-M 설정을 변경하며 성능과 속도를 비교해보고 싶었으나 프로젝트 마감 문제로 16으로만 훈련
- AWS EC2를 활용해 훈련을 진행
- Poly-16 훈련 결과가 MRR 0.811을 기록해 충분하다는 판단 하에 모델 활용 방안 강구를 시작
- 훈련이 끝난 모델의 Attributes를 불러와 함수의 일부로 활용할 수 있다는 것을 알게 됨
- 필요한 함수를 작성하고 파이프 라인을 구축해 결과값을 검증
- 답변 데이터 목록의 임베딩을 미리 산출해 GPU에 로드해두는 것이 핵심이었음
- GPU 과부하 방지를 위해 계산 시에만 Tensor를 GPU 메모리에 배치하는 로직을 추가
- 데모를 위해 터미널에서 작동하는 챗봇 UI를 작성
- 부트캠프 프로젝트 평가 1위를 달성했으나 프로젝트 관리나 모듈화 측면에서 아쉬움이 있었기에 추후 보완을 계획하게 됨

### 참고자료

---

## Poly Encoder

### 프로젝트 개요
- 폴리 인코더 및 리트리벌 시스템을 이해하고 챗봇 구현 방안을 모색한다.

### 개발환경 및 사용기술
- Colab, AWS EC2, Ubuntu, Conda, Git/Github
- PyTorch, Huggingface, SentenceTransformers, Faiss, Pandas, Dialogflow

### 수행사항
- 레퍼런스 및 관련 기술 탐색
- 논문 리뷰 및 세미나 진행
- 노후화된 코드 수정
- 벤치마크를 위한 폴리 인코더 훈련
- SBERT.net 문서 및 예시 학습
- Bi Encoder와 Faiss를 활용한 간단한 챗봇 시스템 구성
- Retrieval Based Chatbot with Poly Encoder 초기 계획 수립

### 회고
- 챗봇 구현에 앞서 Poly Encoder를 위시한 IR System의 개념을 이해하고 관련 기술을 정리해 구체적인 구현 방안과 계획을 수립하기 위해 보낸 한 주
- Papers With Code의 Poly Encoder 페이지를 참조해 논문 리뷰를 진행하는 한편 신뢰할 만한 구현으로 추천받은 chijames/Poly-Encoder를 포크해 코드 분석을 진행
- DSTC 7 데이터로 벤치마크 훈련을 진행하며 노후화된 코드를 보수
- 추천 시스템 성능 지표인 R@K와 MRR에 대해 조사
- SBERT.net의 문서와 예시를 참조해 Sentence Embedding, Similarity Score, Semantic Search, Retrieve and Re-Rank에 대해 학습하고, SentenceTransformers의 Bi/Cross Encoder를 활용한 IR System을 사용해봄으로써 구성 전반에 대해 이해하게 됨
- SentenceTransformers의 Bi Encoder와 Faiss를 활용한 간단한 챗봇 구성 예시를 통해 Poly Encoder를 활용한 챗봇 시스템의 구상을 시작
- Faiss를 사용하며 신속한 Similarity Score 계산을 위한 GPU 연산의 중요성을 깨달음
- 훈련과 검증으로 마무리되었던 지금까지의 프로젝트와는 달리 모델을 시스템 전체의 일부로 사용하는 기술을 처음으로 접하게 됨
- 사전에 Dialogflow를 사용해보며 접했던 기술들의 원리가 무엇인지 이해하게 됨

### 참고자료

---

## Vanila Transformer Implementation

### 프로젝트 개요
- 더 높은 수준의 구현 능력과 모델 활용을 위해 PyTorch를 학습하고 바닐라 트랜스포머를 구현한다.

### 개발환경 및 사용기술
- Colab, WSL2, Ubuntu, Conda, Git/Github
- PyTorch, NumPy

### 수행사항
- 학습 레퍼런스 탐색
- 논문 리뷰 및 세미나 진행
- 파이토치 공식 튜토리얼 학습
- 바닐라 트랜스포머 구현 예시 탐색 및 학습
- 구현에 사용된 클래스와 펑션 정리

### 회고
- 차주 진행 예정으로 있던 Poly Encoder 프로젝트 직전 수행 능력 구비를 위해 진행한 스터디-프로젝트
- 딥 러닝 학습을 이어오며 현재 많은 모델이 PyTorch로 구현되어 있음을 알 수 있었고 활용도를 높히기 위해 관련 학습이 필수라는 생각이 커지고 있던 시점
- Framework의 API 구성을 파악하고 각종 Tensor 조작/연산이 가능해진다면 큰 어려움 없이 PyTorch를 사용할 수 있을 것이라 판단함
- 명확한 지향점을 만들기 위해 바닐라 트랜스포머 구현을 최종 목적으로 설정
- 많은 레퍼런스가 있었지만 단기간 효과적인 학습 성과를 염두에 뒀을 때 공식 튜토리얼의 구성이 가장 설득력 있었음
- 튜토리얼을 진행하며 딥 러닝 기초에 대해 복습하고 다수의 PyTorch 구현 예시를 접함
- 주요 모듈을 파악하며 독해력을 올리는 동시에 바닐라 트랜스포머 구현 예시를 탐색
- 논문과 리뷰 자료를 참고해 트랜스포머의 구조를 파악하며 Tensor Representation이 각각의 블록을 거치며 어떻게 변하는지 추적해봄
- 바닐라 트랜스포머 구현까지는 시간이 부족했으나 구현 예시에 사용된 모듈들의 입출력을 살펴보는 과정을 통해 여러 Tensor 조작/연산을 확인할 수 있었음
- 여타 Python의 클래스처럼 Attributes와 Methods를 이해하는 것으로 모델을 사용할 수 있다는 자신감을 얻게 됨
- 특히 foward 부분에 집중하는 것만으로도 해당 블록이 어떤 작업을 하는지 알 수 있게 된 것은 큰 수확이었음

### 참고자료

---

## Text Summarisation with BART

### 프로젝트 개요
- AI 기반 회의 녹취록 요약 경진대회
- 한정된 데이터와 자원을 활용해 높은 ROUGE 스코어의 요약 모델을 개발한다.

### 모델 / 데이터
- ainize/kobart-news / csebuetnlp/mT5_multilingual_XLSum / google/mt5-small
- 안건별 회의록 및 요약문 2994건(증식 후 247756건)

### 개발환경 및 사용기술
- Colab, Git/Github, WandB
- Pandas, Huggingface, PyTorch

### 수행사항
- 데이터 리포맷
- 데이터 분석 및 시각화
- 모델 선택 및 훈련 코드 작성
- Agenda / Evidence / Summary 클래스 활용 방안 검토
- 훈련 및 추론 / 모델 및 클래스별 ROUGE 스코어 비교
- Gradient Accumulation을 통한 자원 관리
- Mixed Precision Training을 통한 훈련 속도 향상
- Cosine LR Scheduler를 활용한 모델 성능 향상
- Easy Data Augmentation 기법을 사용한 데이터 증식(PLM 활용)
- WandB를 로그를 통한 하이퍼 파라미터 튜닝 실험
- Early Stopping 효과 실험
- 증식 데이터의 효과적인 활용을 위한 훈련 방안 모색
- 각종 제너레이션 메소드 실험
- ROUGE 스코어 기록 경쟁

### 결과 / 성적
- Public 7위(전체 489팀/연습참가)

### 회고
- 다양한 NLP Downstream tasks를 경험해보고자 문서 요약 경진 대회에 참가
- 벤치마크를 할 만한 상위 성적의 베이스라인이 없었기 때문에 Huggingface의 요약 태스크 튜토리얼을 학습하는 것부터 시작
- 요약 데이터로 사전 학습된 모델의 요약 성능이 더 좋다는 정보를 입수해 한국어로 사전 학습된 Pegasus 계열을 검색해봤으나 발견할 수 없었음
- 대안으로 대규모 요약 데이터로 파인튜닝된 모델들의 성능을 ROUGE 스코어로 평가
- Colab 자원을 활용하는 프로젝트였기 때문에 짧은 시간에 효과적인 훈련이 가능했던 ainize/kobart-news로 최종 결정
- 자원 관리를 위해 Gradient Accumulation을 사용하기 시작
- 훈련 데이터의 클래스는 Agenda / Evidence / Summary로 나눠져 있었으며 목표는 Agenda에서 Summary로 요약하는 것이었음
- Evidence는 Agenda에서 추출한 주요 문장으로 구성된 클래스
- 생성 요약의 특성상 Evidence 클래스의 활용이 의미 없다고 판단했으나 Agenda to Evidence 및 Evidence to Summary의 훈련 효율을 시험해봄
- 양자 모두 Agenda to Summary에 비해 낮은 ROUGE 스코어를 기록했기에 폐기 후 데이터 증식으로 방향을 전환
- PLM 모델을 활용한 Easy Data Augmentation 기법 적용 예시를 찾아 데이터 증식에 활용
- RI/SR 기법으로 최초 3배수 증식 이후 연산 속도가 빠른 RD/RS를 사용해 데이터를 80배수 이상으로 증식
- 증식 데이터가 원본의 파생임을 감안해 7500 Step(Iteration) 기준으로 검증 및 체크포인트 생성이 이뤄지도록 설정
- 원본 데이터만 사용한 훈련에서 빠르게 과적합이 일어났던 것과 달리 증식 데이터를 통해 유의미한 성능 향상을 이뤄냈지만 일정 Step 이후부터는 진전이 없음을 발견
- ROUGE 및 Eval/Loss 각각을 기준으로 모델 성능을 시험
- ROUGE 스코어 산출을 위해서는 검증에서 제너레이션 비용을 고려해야 하는데 Eval/Loss 기준으로 체크포인트를 선택했을 때 최고 성적이 나오는 것을 확인
- 훈련 단계에서 ROUGE 스코어 계산이 큰 의미가 없다고 느낌
- Eval/Loss 기준으로 모델을 선택한 뒤 성능 향상을 위해 각종 제너레이션 메소드 실험을 진행
- 실제 요약물의 활용 가능성을 살펴보기 위한 정성 평가 진행 후 Encoder-Decoder 계열 PLM의 성능이 상당하다는 것을 알 수 있었음
- 과거 프로젝트에서 경험한 RNN 계열의 Encoder-Decoder 번역기와 비교시 태스크 차이는 있으나 훈련 방식이 동일하다는 점을 고려했을 때 Transformer와 PLM의 등장으로 NLP 전반의 수행 능력이 크게 높아졌음을 체감

### 참고자료

---

## Text Classification with BERT

### 프로젝트 개요
- Dacon 뉴스 토픽 분류 AI 경진대회
- 한정된 데이터와 자원을 활용해 높은 정확도의 분류 모델을 개발한다.

### 모델 / 데이터
- klue/roberta-base / bert-base-multilingual-uncased / xlm-roberta-base
- 7개 범주 신문기사 제목 45654건(증식 후 101864건)

### 개발환경 및 사용기술
- Colab, Git/Github
- Huggingface, PyTorch, Scikit-Learn, Pandas, hanja, translators, googletrans

### 수행사항
- 데이터 리포맷
- 데이터 분석 및 시각화
- 모델 선택 및 훈련 코드 작성
- 훈련 및 추론 / 모델 간 정확도 비교
- 백 트랜슬레이션 기법을 사용한 데이터 증식(PLM 활용) / 오버 샘플링
- Stratified K-Fold / 로짓 앙상블을 위한 훈련
- 가중치 적용 로짓 앙상블
- 정확도 기록 경쟁

### 결과 / 성적
- Public 19위(전체 418팀/연습참가)

### 회고
- 다양한 NLP Downstream tasks를 경험해보고자 문서 분류 경진 대회에 참가
- 분류 태스크 수행에 효과적인 방법들을 탐색
- 마감 기한을 고려해 사전 시험 성적이 좋았던 3개 모델로 Stratified K-Fold / 로짓 앙상블을 진행하기로 함
- 앙상블 연산을 구현하기 위해 BERT의 출력 로짓을 이해하는 과정을 거침
- 처음으로 앙상블을 기획하고 훈련부터 추론까지 모두 직접 구현한 프로젝트였기 때문에 이론과 실제가 어떻게 연결되는지 배울 수 있는 좋은 기회였음
- Subword 방식의 SentencePiece 토크나이저와 Encoder 계열 PLM을 활용한 문서 분류 모델 훈련에 있어 성능과 관련되는 주요 요인을 조사
- 학습 목표에 해당하지 않는 샘플 혹은 노이즈를 제거하는 것과 토크나이저가 인식하지 못 하는 토큰을 대체해주는 것이 중요하다는 것을 배움
- 한자 캐릭터가 UNK 토큰으로 분류되는 경우가 잦은 것을 감안해 hanja 패키지 활용을 검토
- EDA, Back Translation, TextAttack과 같은 Text Data Augmentation에 대해 알게 됨
- 데이터의 도메인에 적합한 증식 방법을 잘 선택하는 것이 중요하고 경우에 따라 효과가 미미하거나 없을 수도 있다는 것을 배움
- translators 패키지를 활용해 검색 엔진의 번역기 API로 Back Translation 증식을 진행했으나 지나치게 잦은 요청으로 차단되는 상황이 발생
- M2M-100 모델을 활용해 Back translation 증식을 수행하고 증식 데이터의 노이즈를 제거
- 앙상블 효과를 높히기 위해 모델별 로짓에 가중치를 변경하며 실험을 진행
- 기획부터 구현까지 각각의 판단들이 모여 하나의 결과물을 만들어내는 만큼 다양한 가능성에 대해 생각하고 타당성을 검토하며 프로젝트를 진행해야 한다는 것을 배움

### 참고자료

---

## Poetry Generator with GPT2

### 프로젝트 개요
- 시 생성 언어모델 개발을 위해 웹 크롤링한 데이터를 활용, GPT2를 파인튜닝한다.

### 모델 / 데이터
- skt/kogpt2-base-v2
- 웹사이트 시 사랑 시의 백과사전 시인의 시 106371건

### 개발환경 및 사용기술
- Colab, Git/Github
- BeautifulSoup, Pandas, RE, Huggingface, PyTorch

### 수행사항
- 웹 크롤링 및 텍스트 데이터 정제
- GPT-2 토크나이저로 토큰화 및 정수 인코딩 / SentencePiece 토크나이저 학습
- 훈련 코드 작성 및 훈련
- 문장 생성 종료를 위한 스페셜 토큰 추가
- 각종 제너레이션 메소드 실험

### 회고
- 처음으로 Transformer PLM을 활용해본 프로젝트
- 당시에는 Masked LM보다는 Causal LM의 활용 방안이 더 직관적이었기 때문에 한국어로 사전 학습 된 GPT 계열의 모델을 탐색함
- SKT에서 사전 학습한 KoGPT-2를 발견하고 Transformer PLM 허브인 Huggingface를 활용해야 함을 인지
- 창작 활동에 도움을 주는 모델을 개발하고자 데이터를 물색했으나 실패
- 웹 크롤링을 통해 10만건 이상의 시 데이터를 확보
- Raw 데이터의 경우 유형화된 노이즈가 상당 수 존재했기에 정규표현식을 활용해 제거
- Huggingface API 구성을 파악한 뒤 Transformers, Datasets, Tokenizers를 활용해 훈련 코드를 작성
- 훈련이 끝난 모델의 생성 문장에서 동일 단어 반복을 줄이기 위해 각종 제너레이션 메소드를 이해한 뒤 적용
- 생성된 시가 종결될 수 있도록 EOS 토큰을 추가하여 재훈련
- 훈련 성과 측정을 위해 파인튜닝 전후 Perplexity를 산출해 비교
- 기발한 표현의 문장들이 생성되는 것을 기대했으나 최종 결과물로 산출한 문장들은 복고풍의 서정시인 경우가 많았음
- 좋은 결과물을 얻기 위해서는 좋은 데이터를 사용해야 한다는 점을 재차 확인

### 참고자료

---

## LSTM Seq2Seq with Attention

### 프로젝트 개요
- 영한 번역기 개발을 위해 LSTM Encoder-Decoder with Attention 모델을 훈련하고 BLUE 스코어를 측정한다.

### 모델 / 데이터
- LSTM Encoder-Decoder with Attention
- AIHub 한국어-영어 번역(병렬) 말뭉치 구어체 20만건

### 개발환경 및 사용기술
- Colab
- Tensorflow, Pandas, RE

### 수행사항
- 성능 향상을 위한 모델 수정
- 데이터 확보 및 정제
- 토큰화 및 코퍼스 사전 구축 / 훈련 및 추론
- 추가 데이터 확보 / 시퀀스 역순 입력 / 훈련 및 추론
- BLUE 스코어 기록 경쟁

### 결과 / 성적
- BLUE 스코어 기록 경쟁 1위

### 회고
- RNN 계열 수업 후 Encoder-Decoder 모델의 원리를 익히고 Attention 기작의 효과를 확인하기 위해 진행한 프로젝트
- Attention이 포함되지 않은 LSTM Seq2Seq 모델의 레이어 구성과 하이퍼 파라미터 수정을 수차례 진행했으나 번역이라고 느껴질 만한 결과물을 만들어내지 못 함
- 동일한 데이터로 Attention을 포함한 모델이 월등히 우수한 결과물을 만들어내는 것을 확인
- Encoder-Decoder 모델에서 시퀀스 역순 입력으로 성능 향상이 가능하다는 정보를 입수해 직접 비교 실험을 실시했으나 당시 BLUE 스코어 이해 부족으로 정성 평가만 진행
- 추가 데이터로 최종 훈련을 진행하고 기록 경쟁에서 1위를 차지
- 함수형 모델 작성과 수정에 있어서 어려움이 많았기 때문에 추후 복습과 재학습의 필요성을 느낌

### 참고자료

---

## Text Mining with ML

### 프로젝트 개요
- 웹툰 독자의 경향을 파악하기 위해 수치 데이터를 웹 크롤링 후 분석 및 시각화를 진행한다.
- 우크라사태가 한국 경제에 미치는 영향을 파악하기 위해 웹 크롤링한 신문기사에 워드 클라우드, 토픽 모델링 기법을 활용한다.
- 로지스틱 리그레션을 활용해 우크라사태 이전/이후 신문기사 내용의 감성 분석을 진행한다.

### 모델 / 데이터
- TF-IDF / LDA / Logistic Regression
- 네이버 웹툰 장르별 회차별 독자 참여 수치 19622건
- 우크라사태 이전/이후 신문기사 제목 및 본문 2912건

### 개발환경 및 사용기술
- Colab
- BeautifulSoup, Pandas, RE, KoNLPy, Scikit-Learn, Gensim, Matplotlib, Seaborn

### 수행사항
- 웹 크롤링 및 텍스트 데이터 정제
- 데이터 분석 및 시각화
- 토큰화 및 코퍼스 사전 구축
- 불용어 사전 및 사용자 사전 구축
- TF-IDF 워드 클라우드
- LDA 토픽 모델링
- 바이그램 / 트라이그램 적용
- 응집도 / 복잡도 기준 최적 에폭 및 토픽 갯수 설정
- 로지스틱 리그레션을 활용한 감성 분석

### 회고
- 웹 크롤링과 ML 기초를 배운 후 유의미한 적용을 위해 진행한 프로젝트
- BeautifulSoup을 사용해 목표 범위의 데이터를 크롤링하는 코드를 작성
- 웹 크롤링을 기법으로 단기간에 방대한 데이터 확보가 가능함을 알게 되었고, 재사용 가능한 코드 작성의 중요성에 대해 배울 수 있었음
- 정규표현식을 활용해 불필요한 문자 및 패턴이 있는 노이즈를 찾아 제거
- Gensim/wordcloud 사용으로 구현 난이도는 높지 않았으나 처음부터 만족스러운 결과를 얻을 수 없었음
- 수행하려는 태스크가 이미 패키지로 구현되어 있는 경우라도 다양한 조건으로 실험을 진행해야 보다 완성도 있는 결과물을 산출할 수 있음을 배움
- KoNLPy의 Komoran, Mecab, Okt의 성능을 직접 시험해보고 사용자 사전 등록이 가능한 Komoran을 토크나이저로 선택
- `18년 논문인 “텍스트마이닝을 위한 한국어 불용어 목록 연구”를 참조해 불용어 사전 구축을 시작하고 고유명사 중심으로 사용자 사전을 구축
- 사용자 사전 없이 바이그램 / 트라이그램 결과 시험을 해봤으나 유의미한 성과는 없었음
- 사전 구축에 최대한의 노력을 기울였으며 이후 최적 에폭 및 토픽 갯수 설정을 위한 실험을 진행
- 실험 결과 보존이 되지 않은 상황에서 재현조차 불가능한 경우가 있었는데 답이 정해져 있지 않은 만큼 경험적으로 접근하며 기록을 남겨야 함을 깨달음
- 같은 하이퍼 파라미터 설정으로 다른 결과가 도출되는 것을 방지하기 위해 랜덤스테이트(Seed)를 고정해야 함을 알게 됨

### 참고자료

---

## Object Detection with YOLOv5

### 프로젝트 개요
- 폐기물 객체 인식 모델을 개발하기 위해 AIHub의 생활 폐기물 데이터를 활용해 YOLOv5를 훈련한다.

### 모델 / 데이터
- YOLOv5
- AIHub 생활 폐기물 이미지 일부 / TACO 데이터셋 일부

### 개발환경 및 사용기술
- Colab, WSL2, Ubuntu, Conda, WandB
- YOLOv5, PyTorch, OpenCV, Pillow, NumPy, Matplotlib

### 수행사항
- YOLOv5 모델의 규격에 맞춰 어노테이션을 리포맷
- 이미지 크기 조절, 중앙 배치, 패딩, 제로 센터링
- WandB 로그를 통한 훈련 과정 모니터링
- 추가 데이터 후속 훈련 및 추론
- 카메라 연동을 통해 실제 환경에서 시험

### 결과 / 성적
- 부트캠프 객체 인식 팀 프로젝트 1위

### 회고
- YOLOv3 실습 이후 객체 인식 어플리케이션의 기획 및 구현을 시도한 프로젝트
- 객체 인식 모델의 성능 평가 지표인 IOU, mAP에 대해 학습
- 평가 지표의 개념을 토대로 객체 인식 태스크를 이해
- YOLO 시리즈의 Backbone-Neck-Head 구조에 대해 세미나를 진행
- Feature Extraction을 수행하는 Backbone의 중요성이 가장 크다는 것을 배움
- 훈련을 진행하는 과정에서 CUDA out of memory RuntimeError를 처음으로 경험
- Colab 환경에서 훈련이 가능한 체크포인트(YOLOv5m)와 배치 사이즈를 선택
- 최초 훈련 후 추가 데이터로 후속 훈련을 진행했으나 오히려 정확도가 낮아지는 결과가 나옴
- WandB Evaluation Log를 확인해 어노테이션 리포맷의 오류를 발견한 뒤 수정
- 훈련 후 카메라 연동을 통해 모델을 실제 환경에서 시험해봤으나 기대에 못 미치는 결과를 얻음
- 훈련/검증 데이터의 구성에 사용 환경의 맥락이 반영되어 있어야 의도했던 추론 결과를 얻을 수 있음을 배울 수 있었음

### 참고자료

---

## Image Classification with ResNet

### 프로젝트 개요
- 재활용품 분류를 위한 모델을 개발하기 위해 이미지를 웹 크롤링한 후 ResNet을 파인튜닝한다.

### 모델 / 데이터
- ResNet
- 웹 크롤링 재활용품 이미지 데이터 2000건

### 개발환경 및 사용기술
- Colab, WSL2, Ubuntu, Conda
- Tensorflow, Selenium, NumPy, Pillow, OpenCV, Pandas

### 수행사항
- 웹 크롤링
- 이미지 크기 조절, 중앙 배치, 패딩, 제로 센터링
- 데이터 증식
- LR Scheduler 및 Early Stopping 사용
- 파인튜닝 및 추론

### 회고
- 기본적인 이미지 전처리와 Tensorflow를 활용한 이미지 데이터 증식을 실습하기 위해 진행한 프로젝트
- 제 각기 다른 비율의 이미지를 훈련 데이터로 사용하는 것은 추론 단계에서 오류를 일으킬 가능성이 있다는 생각이 들어 관련 정보를 탐색
- 일반적인 전처리 과정인 중앙 배치와, 패딩, 제로 센터링에 대해 알게 됨
- 또한 훈련 데이터 부족을 만회하기 위해 각종 증식 기법을 적용해봄
- 데이터의 도메인을 고려한 증식 기법이 아닌 경우 오히려 악영향을 미칠 수 있다는 것을 알게 됨

### 참고자료
- 각종 API 문서 및 이미지 분류 튜토리얼
