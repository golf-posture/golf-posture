# 베이스라인 스냅샷 - 2026-09-25

`documents/feedback_improvement_plan_v2.md` 1주차 작업. v2 항목 1~4(Mahalanobis/다이내믹스/클러스터링/문구) 변경 전, 현재 파이프라인 그대로의 출력을 고정해 이후 변경과 diff 비교할 근거 자료.

## 생성 조건

- 커밋 해시: `382842d9dc48bdd40c925ad59ed6a2b2bbdfc8de` (브랜치 `feature/feedback-v2` 분기 지점, `main`)
- 사용 통계: `data/pro_feature_stats_reliability.json` (이 폴더에 동일 파일 복사본 포함)
- 사용 reference skeleton: `data/pro_reference_skeletons.json`
- SwingNet checkpoint: `event_detection/models/swingnet_1800.pth.tar` (mock-events 아님, 실제 검출)
- handedness: `right` (전 영상 동일 적용, 실제 왼손잡이 여부는 별도 확인 안 함)
- pose backend: `hybrid`

## 입력 샘플

`raw_videos/`의 6개 영상 전부:

```text
longform.mp4
shorts.mp4
shorts2.mp4
test1.mp4
test2.mp4
test3.mp4
```

## 실행 커맨드

```bash
pixi run python -m pipeline.run_analysis \
  --video raw_videos/<name>.mp4 \
  --checkpoint event_detection/models/swingnet_1800.pth.tar \
  --pro-stats data/pro_feature_stats_reliability.json \
  --reliability-stats data/pro_feature_stats_reliability.json \
  --reference-skeletons data/pro_reference_skeletons.json \
  --output outputs/baseline_snapshot/<name> \
  --handedness right \
  --pose-backend hybrid
```

## 결과 요약

6개 영상 전부 `errors: []`, 8/8 이벤트 pose 검출 성공. 실패 케이스 없음.

## 보관 파일

각 `<name>/` 폴더에 다음 4개 파일만 보관 (skeleton/frame 이미지는 용량 문제로 제외):

```text
<name>_result.json    # z_score/grade 포함 전체 분석 결과
<name>_summary.txt     # 텍스트 요약
feedback_table.csv     # 8이벤트 x 5feature 평가표
feedback_ko.txt        # 한국어 피드백 문구
```

## 사전 환경 메모

`event_detection/model.py`가 `event_detection/mobilenet_v2.pth.tar`(루트)를 요구하는데 로컬에는 `event_detection/models/mobilenet_v2.pth.tar`에만 있어, 이 스냅샷 생성 전 README에 문서화된 위치로 복사해 두었음(`cp event_detection/models/mobilenet_v2.pth.tar event_detection/mobilenet_v2.pth.tar`, 가중치 파일이라 git 추적 대상 아님).

## 이후 사용 방법

2~4주차 변경(Mahalanobis, 다이내믹스, 클러스터링, 문구 계층) 완료 후 동일 6개 영상 + 동일 커맨드로 재실행해 `<name>_result.json`의 `grade`/`z_score`(또는 신규 필드)와 `feedback_ko.txt` 문구를 이 폴더 내용과 diff 비교한다.
