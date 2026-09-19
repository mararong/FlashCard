# English Vocabulary Builder

`wordfreq`의 빈도 목록에서 영어 단어 후보를 자동 생성하고, WordNet 사전 정보와
Zipf 빈도 및 Ollama의 난이도 평가를 결합해 플래시카드용 JSON 데이터베이스를 만든다.

## 환경 설정

Python 3.10 이상을 권장한다. 프로젝트 루트에서 가상 환경을 만들고 의존성을 설치한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader wordnet
```

WordNet 다운로드 위치가 기본 NLTK 경로가 아니라면 `NLTK_DATA` 환경 변수를 해당
디렉터리로 설정한다.

## Ollama 설정

1. Ollama를 설치하고 서버를 실행한다.
2. 사용할 모델을 받는다. 기본 모델은 `qwen2.5:7b`이다.

```powershell
ollama pull qwen2.5:7b
ollama serve
```

모델은 `--model` 또는 `OLLAMA_MODEL`, 서버 주소는 `--ollama-url` 또는
`OLLAMA_HOST`로 바꿀 수 있다. 기본 주소는 `http://localhost:11434`이며 AI
temperature는 일관성을 위해 0으로 고정한다.

## 실행

소규모 테스트:

```powershell
python scripts/build_english_vocabulary.py --limit 20
```

전체 데이터(상위 30,000개 후보에서 최대 10,000개):

```powershell
python scripts/build_english_vocabulary.py
```

주요 옵션:

```text
--candidate-count 30000  wordfreq에서 읽을 후보 수
--limit 10000           저장 목표 단어 수
--batch-size 30         AI 요청당 단어 수
--max-senses 8          단어당 사용할 주요 WordNet sense 수
--model qwen2.5:7b      Ollama 모델
--ollama-url URL        Ollama 서버 주소
--resume                기존 결과에서 재개(기본 동작)
--no-resume             출력이 없는 경우에만 새로 시작
--reset                 기존 두 JSON을 비우고 처음부터 시작
```

각 AI batch가 성공할 때마다 두 JSON을 원자적으로 저장한다. 중간 종료 후 같은
명령을 다시 실행하면 `english_vocabulary.json`에 이미 있는 단어는 건너뛴다.
AI 연결/응답 오류는 기본 3회 재시도한 뒤 명확히 출력하고, 처리 완료 데이터는
보존하며 해당 batch를 `retryable: true`로 기록한다. 설정을 고친 뒤 재실행하면
그 batch부터 다시 처리한다.

> `--reset`은 기존 생성 결과를 빈 배열로 교체한다. 의도적으로 처음부터 다시
> 만들 때만 사용한다.

## 출력 데이터

- `data/english/english_vocabulary.json`: 단어, 품사, 정의, 주요 sense, 예문,
  동의어/반의어, Zipf와 난이도 점수. JSON 배열이다.
- `data/english/rejected_words.json`: WordNet 정보가 없거나 AI batch가 실패한
  후보와 단계, 사유, 재시도 가능 여부.

난이도 총점은 Python이 `frequency + abstractness + semantic_complexity +
form_complexity + register`로 계산한다. 0~3은 Level 1, 4~6은 Level 2,
7~10은 Level 3이다. AI는 frequency를 평가하지 않는다.

## 테스트

```powershell
python -m unittest discover -s tests -v
```
