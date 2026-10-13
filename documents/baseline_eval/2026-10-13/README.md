# 베이스라인 스냅샷 - 2026-10-13 (추적 추출)

`2026-09-25` 스냅샷(정지 프레임 추출, v2 변경 전)에 이어, **추적 추출이 기본값이 된 이후** 파이프라인 출력을 고정한 스냅샷. 이후 변경과 diff 비교하는 근거 자료.

## 생성 조건

- 커밋 해시: `7d6ec5755e714a4bd0b7d2a9199f3a70a1dbe2ef` (브랜치 `feature/feedback-v2-week2`) + 작업 트리의 미커밋 변경(작업 전부터 있던 `pipeline/generate_feedback_report.py` 수정 포함). 리포트 문구는 이 수정이 반영된 상태다.
- 추출 방식: `--extraction tracked` (기본값). 영상 모드 추적 + 사용자 팔꿈치 QC.
- 기준 통계: `--reference legacy` (기본값) → `data/pro_feature_stats_reliability.json`, `data/pro_feature_covariance.json`, `data/pro_dynamics_stats.json` (2026-09-25 스냅샷과 동일한 기준 파일)
- SwingNet checkpoint: `event_detection/models/swingnet_1800.pth.tar` (실제 검출)
- handedness: `right` (전 영상 동일 적용)
- pose backend: `mediapipe` (`hybrid`는 추적 추출이 적용되지 않으므로 사용하지 않음)

## 2026-09-25 스냅샷과 비교할 때

- 기준 통계 파일은 같다. 달라진 것은 (1) 추출 방식(정지 → 추적), (2) pose backend(`hybrid` → `mediapipe`), (3) 그 사이 v2 1~3주차 코드 변경(Mahalanobis, 다이내믹스, 문구)이다. 따라서 등급 차이는 이 셋이 섞여 있다.
- 같은 6개 영상에서 정지 방식 대비 등급 차이는 240개 항목 중 37개(15%)였다(작업 기록 3.2절, 두 방식 모두 `mediapipe` 백엔드로 비교한 값).

## 입력 샘플

`raw_videos/`의 6개 영상 전부: `longform`, `shorts`, `shorts2`, `test1`, `test2`, `test3`.

## 실행 커맨드

```bash
pixi run python -m pipeline.run_analysis \
  --video raw_videos/<name>.mp4 \
  --checkpoint event_detection/models/swingnet_1800.pth.tar \
  --output outputs/baseline_snapshot_tracked/<name> \
  --handedness right \
  --pose-backend mediapipe
```

## 결과 요약

6개 영상 전부 `errors: []`, 결과 JSON에 `extraction: tracked`, `reference: legacy` 기록.

## 보관 파일

각 `<name>/` 폴더에 4개 파일만 보관(이미지는 용량 문제로 제외):

```text
<name>_result.json    # z_score/grade 포함 전체 분석 결과
<name>_summary.txt     # 텍스트 요약
feedback_table.csv     # 8이벤트 x 5feature 평가표
feedback_ko.txt        # 한국어 피드백 문구
```
