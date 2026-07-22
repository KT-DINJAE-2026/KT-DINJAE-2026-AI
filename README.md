# AI
KT 디인재 프로젝트 AI 레포지토리

## 구조
- `docs/` — 회의록, 데이터 스펙 문서
- `schema/` — 백엔드와 합의한 학습 데이터 예시 스키마
- `src/train_models.py` — 모델A(입석여부)/모델B(입석시간) 학습
- `src/inference.py` — 학습된 모델로 예측 결과를 API 응답 형태로 변환

## 실행
\`\`\`bash
pip install -r requirements.txt
python src/train_models.py
python src/inference.py
\`\`\`
