# 🤖 LLM Assistant

> Gemini 기반 RAG · LangChain · LangGraph · FastAPI 파이프라인 구현

## 📌 프로젝트 개요

Gemini API를 활용한 LLM 서비스 개발 전 과정을 단계적으로 구현한 프로젝트입니다.
단순 API 호출 수준을 넘어 LLM 시스템의 내부 동작 원리를 직접 구현하며 체득하는 방식으로 진행하였습니다.

- **수행 형태**: 개인 프로젝트
- **수행 기간**: 2026.01 ~ 2026.03
- **Demo**: [HuggingFace Spaces](https://huggingface.co/spaces/sngdmtdkw-02/llm-assistant)

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|------|------|
| LLM | Gemini API (gemini-2.0-flash-lite) |
| Framework | LangChain, LangGraph |
| RAG | ChromaDB, ko-sroberta-multitask |
| API | FastAPI, Uvicorn |
| UI | Gradio |
| 배포 | HuggingFace Spaces, Docker |
| 버전 관리 | Git Flow |

---

## 📂 프로젝트 구조
```
llm-assistant/
├── assistant/          # 1단계: LLM 기초
├── tools/              # 2단계: Function Calling
├── prompts/            # 2단계: Prompt Engineering
├── rag/                # 3단계: RAG 파이프라인 직접 구현
├── langchain_app/      # 4단계: LangChain 애플리케이션
├── agent/              # 5단계: LangGraph AI Agent
├── core/               # 6단계: 시스템 엔지니어링
├── evaluation/         # 6단계: RAG 품질 평가
├── api/                # 7단계: FastAPI REST API
├── app.py              # Gradio UI (HuggingFace Spaces)
└── requirements.txt
```

---

## 🗂️ 단계별 구현 내용

### 1단계 — LLM 기초 `v1.0.0`
- Gemini API 기반 멀티턴 대화 시스템 구현
- 스트리밍 응답, 토큰 카운터, 대화 히스토리 관리

### 2단계 — Function Calling + Prompt Engineering `v2.0.0` `v2.1.0`
- 날씨 조회, 계산, 시간 조회 도구 연동
- CoT, Few-shot, Role Prompting 등 다양한 기법 실험

### 3단계 — RAG 시스템 `v3.0.0`
- PDF 파싱 → 청킹 → 임베딩 → ChromaDB 벡터 검색 파이프라인 직접 구현
- ko-sroberta-multitask 기반 한국어 문서 검색

### 4단계 — LangChain 애플리케이션 `v4.0.0`
- LCEL 파이프라인, RunnableWithMessageHistory 활용
- Window Memory / Summary Memory 비교 구현

### 5단계 — LangGraph AI Agent `v5.0.0`
- StateGraph, ToolNode, tools_condition 활용 ReAct Agent 구현
- 멀티 도구 병렬 호출 및 조건부 엣지 설계

### 6단계 — LLM 시스템 엔지니어링 `v6.0.0`
- Pydantic 기반 Config 관리, 구조화 로깅, LRU 캐싱
- LLM-as-a-Judge 기반 RAG 품질 평가 (종합 점수 0.89 / 1.0)

### 7단계 — 서비스 배포 `v7.0.0`
- FastAPI REST API 서버 구축 (Chat · RAG · Agent 엔드포인트)
- Docker 컨테이너화
- Gradio UI 제작 및 HuggingFace Spaces 배포

---

## 📊 성과

- RAG 평가 종합 점수 **0.89 / 1.0** 달성 (LLM-as-a-Judge 기반)
- LLM 직접 구현 → LangChain → LangGraph 단계적 추상화 수준 체득
- Git Flow 기반 체계적 버전 관리 **(v1.0.0 → v7.0.0)**

---

## 🚀 실행 방법

### 환경 설정
```bash
pip install -r requirements.txt
```

### .env 설정
```
GEMINI_API_KEY=your_api_key
DEFAULT_MODEL=gemini-2.0-flash-lite
```

### Gradio UI 실행
```bash
python app.py
# → http://localhost:7860
```

### FastAPI 서버 실행
```bash
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs
```