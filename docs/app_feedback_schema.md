# 앱용 피드백 JSON 스키마 (`feedback_app.json`, schema_version 2)

분석이 끝나면 `outputs/<이름>/feedback_app.json`이 만들어집니다. 앱의 "분석 결과 상세" 화면을 이 파일 하나로 그릴 수 있게 정리한 것입니다. 만드는 코드는 `pipeline/app_feedback.py`의 `build_app_payload`이고, 문구와 기준값은 같은 파일 위쪽 상수에서 바꿉니다.

기존 `feedback_ko.txt`(상세)와 `feedback_user.txt/json`(일반 사용자용 텍스트)은 이 파일과 별개로 그대로 만들어집니다. 앱 JSON을 만드는 데 실패해도 그 파일들은 영향을 받지 않습니다.

## 1. 화면과 필드 대응

| 화면 요소 | JSON 경로 |
|---|---|
| 자세 점수 (예: 6.8 / 10) | `summary.posture_score` |
| 스윙 리듬 (예: 2.8 : 1) | `summary.tempo.text`, `summary.tempo.caption` |
| 분석 불확실 안내 | `notice` (없으면 `null`) |
| 자세 진단 탭: 문제점 카드 (3~5개, 점수 낮은 순) | `issues[]` |
| 카드 제목 ("문제 이벤트 - 문제점 요약") | `issues[].title` |
| 카드 부제 (피드백 요약) | `issues[].subtitle` |
| 카드 하단 이미지 | `issues[].media.keyframe` (+ `skeleton_compare`, `keyframes`) |
| 카드 자세한 피드백 | `issues[].body[]` |
| 점수 막대와 상태 라벨 | `issues[].score`, `issues[].status` |
| 문제 없음 화면 | `state == "no_issues"`, `empty_state` |
| 풀스윙 탭: 영상과 8개 이벤트 버튼 | `events[]` (`frame_index`로 이동) |
| 칭찬 탭 | `praise[]` |

## 2. 화면 상태

| 조건 | `state` |
|---|---|
| 문제점 카드가 1~5개 | `issues` (개수는 `issues.length`) |
| 문제점 카드가 없음 | `no_issues` (`empty_state`에 문구가 들어옵니다) |

문제 후보가 모자라도 개수를 채우지 않습니다. 분석이 불확실하면 `reliability`가 `low`가 되고 `notice`에 안내 문구가 들어옵니다. 카드는 그대로 보여주고 안내만 띄우면 됩니다. `notice.code`는 `tempo_excluded`, `many_unavailable`, `mock_events` 중 하나입니다.

## 3. 필드 정의

**최상위**

| 필드 | 설명 |
|---|---|
| `schema_version` | 정수. 필드를 지우거나 뜻을 바꿀 때만 올립니다. |
| `meta` | `{handedness, mock_events}` |
| `state`, `reliability` | 위 표 참고. `reliability`는 `normal` 또는 `low` |
| `notice` | `{code, message}` 또는 `null` |
| `empty_state` | `{title, message}` (`no_issues`일 때만) |
| `analysis_notes` | 하단 안내 문구 배열 |

**`summary`**

| 필드 | 설명 |
|---|---|
| `posture_score` | 0~10, 소수 첫째 자리. 분석 항목이 없으면 `null` |
| `posture_score_basis` | 산식 이름. 현재 `within_pro_ratio` (프로 범위 안 항목 비율 x 10, 임시) |
| `score_inputs` | `{within_pro, analyzed}` 점수 계산에 쓴 개수. 화면에 보여줄 값은 아닙니다. |
| `tempo` | `{status, ratio, text, caption, pro_mean}`. `status`가 `excluded`이면 `ratio`는 `null`, `text`는 `"-"` (측정이 불확실한 경우) |

**`issues[]`** (최대 5개, 점수가 낮은 카드부터)

| 필드 | 설명 |
|---|---|
| `rank` | 1부터 시작 |
| `id` | `elbow`, `spine`, `weight_shift` 등 |
| `score` | 0~100. 낮을수록 프로와 많이 다릅니다. |
| `status` | `{code, label}`: `priority` "먼저 확인해요", `adjust` "조금 더 안정적으로", `recheck` "한 번 더 확인해요" |
| `event` | `{key, label}` 카드의 대표 이벤트 |
| `title` | "이벤트 + 문제점 요약". 방향 표현은 쓰지 않습니다. |
| `topic` | `title`에서 이벤트를 뺀 요약 |
| `subtitle` | 피드백 한 줄 요약 |
| `body[]` | 자세한 피드백. `{kind, text}`이고 `kind`는 `change`(구간 변화 설명), `advice`(어떻게 하면 되는지), `caution`(오차 가능성 안내). 없는 종류는 빠집니다. |
| `stats[]` | 선택. 내 값과 프로 값 비교 `{label, at, value_text, pro_text, range_text, tail_pct}`. 화면에서 쓰지 않아도 됩니다. |
| `sources[]` | 이 카드를 만든 분석: `angle`(이벤트별 각도), `motion`(동작 지표), `dynamics`(이벤트 사이 변화량) |
| `media` | `{keyframe, skeleton_compare, keyframes[]}` 첫 이벤트의 이미지와 관련 이벤트 이미지 목록 |

**`praise[]`** (칭찬 탭, 강한 칭찬부터 최대 8개)

`{id, title, subtitle, score, event, media}`. 문제점 카드와 같은 모양이고, 문제점으로 나온 부위는 칭찬에 나오지 않습니다.

**`events[]`** (풀스윙 탭, 항상 8개)

`{key, label, frame_index, issue_ids, media}`. 이벤트 버튼 이름은 어드레스, 토업, 백스윙, 탑, 다운스윙, 임팩트, 팔로스루, 피니시입니다. `issue_ids`는 그 이벤트가 대표 이벤트인 문제점 카드의 `id`입니다.

## 4. 문제점 카드가 만들어지는 방식

- 후보는 세 군데에서 모읍니다. 이벤트별 각도(팔꿈치, 상체 기울기 부위별), 동작 지표 7개, 스윙 다이내믹스(두 이벤트 사이 변화량, 팔꿈치와 상체 기울기에 합쳐짐)입니다.
- 신뢰도가 낮은 항목은 카드가 되지 않습니다. 어깨선과 골반선 기울기는 카메라 영향이 커서 카드로 올리지 않습니다.
- 카드 점수는 `100 - 30 x |z|`(0~100)이고, 카드를 이루는 항목 중 가장 큰 |z|로 계산합니다. |z| 1이면 70점, 2이면 40점입니다.
- 상태 라벨은 점수로 정합니다. 40 미만이면 `priority`, 55 미만이면 `adjust`, 그 외는 `recheck`입니다. 기준값은 임시입니다.
- 각도는 평균과 표준편차, 동작 지표와 다이내믹스는 중앙값과 보정 MAD가 기준이라, 같은 점수라도 심각도가 완전히 같지는 않습니다. 실제 영상을 보면서 보정합니다.

## 5. 문구 규칙

- 제목은 방향이 없는 중립 표현입니다. 방향(예: "들림", "펴짐")이 실제 동작과 맞는지 코치 확인이 끝나지 않았습니다.
- `subtitle`은 한 줄 요약이고, 자세한 설명은 `body[]`에 있습니다. 한 줄 길이는 화면 예시 기준 약 20~30자입니다.
- 전문 용어(z-score, 신뢰도 등급)는 어떤 필드에도 내려보내지 않습니다.

## 6. 백엔드 참고

- `media.*` 값은 분석을 돌린 서버의 파일 경로입니다. 앱에서 보려면 업로드하거나 URL로 바꿔서 내려주세요.
- 이벤트 시간은 `frame_index / fps`로 계산합니다. 분석 결과에 fps가 없어서 영상 메타데이터의 fps를 사용하세요.
- 값이 없는 필드는 키를 빼지 않고 `null`, 빈 배열로 내려갑니다.
- 필드를 추가하는 것은 `schema_version`을 올리지 않습니다. 앱은 모르는 필드를 무시하면 됩니다.

## 7. 보류와 한계

- 전문가용 / 일반용 피드백 탭 분리는 보류했습니다. 코치 검증 전이라 이름을 붙이기 이르다고 판단했고, 필요하면 `stats[]`를 접이식 "상세 수치"로 보여줄 수 있습니다.
- "프로 범위 안에 N개" 문장은 화면에 쓰지 않습니다. `score_inputs`는 자세 점수를 계산하는 값입니다.
- 자세 점수 산식은 임시입니다. 피드백이 완성된 뒤 정합니다. 값만 바뀌고 필드는 그대로입니다.
- 이벤트 위치 오차가 큰 영상에서는 다운스윙~임팩트 구간 팔 각도가 흔들릴 수 있습니다. 이 구간 카드에는 `caution`이 붙습니다.
- 정면 영상 기준의 2D 추정입니다.

## 8. 예시 (문제 3개, 칭찬과 이벤트는 일부만 표시)

```json
{
  "schema_version": 2,
  "meta": {
    "handedness": "right",
    "mock_events": false
  },
  "state": "issues",
  "reliability": "normal",
  "notice": null,
  "summary": {
    "posture_score": 9.1,
    "posture_score_basis": "within_pro_ratio",
    "score_inputs": {
      "within_pro": 43,
      "analyzed": 47
    },
    "tempo": {
      "status": "ok",
      "ratio": 2.8,
      "text": "2.8 : 1",
      "caption": "백스윙 : 다운스윙",
      "pro_mean": 3.9
    }
  },
  "issues": [
    {
      "rank": 1,
      "id": "elbow",
      "score": 22,
      "status": {
        "code": "priority",
        "label": "먼저 확인해요"
      },
      "event": {
        "key": "Impact",
        "label": "임팩트"
      },
      "title": "임팩트 팔꿈치 각도 차이",
      "topic": "팔꿈치 각도 차이",
      "subtitle": "다운스윙~임팩트에서 양쪽 팔꿈치 각도가 프로와 달라요.",
      "body": [
        {
          "kind": "advice",
          "text": "슬로모션으로 프로 영상과 나란히 비교해 보세요."
        },
        {
          "kind": "caution",
          "text": "이 구간은 동작이 빨라 분석 오차가 생기기 쉬워요. 다른 각도에서 한 번 더 촬영해 비교해 보면 더 정확해요."
        }
      ],
      "stats": [
        {
          "label": "타깃 쪽 팔꿈치",
          "at": "임팩트",
          "value_text": "100°",
          "pro_text": "95°",
          "range_text": null,
          "tail_pct": null
        },
        {
          "label": "뒤쪽 팔꿈치",
          "at": "임팩트",
          "value_text": "100°",
          "pro_text": "95°",
          "range_text": null,
          "tail_pct": null
        }
      ],
      "sources": [
        "angle"
      ],
      "media": {
        "keyframe": "frames/5.jpg",
        "skeleton_compare": "skeletons/5.jpg",
        "keyframes": [
          "frames/5.jpg"
        ]
      }
    },
    {
      "rank": 2,
      "id": "spine",
      "score": 34,
      "status": {
        "code": "priority",
        "label": "먼저 확인해요"
      },
      "event": {
        "key": "Top",
        "label": "탑"
      },
      "title": "탑 상체 기울기 차이",
      "topic": "상체 기울기 차이",
      "subtitle": "어드레스에서 탑으로 가는 동안 상체 기울기 변화가 프로와 달라요.",
      "body": [
        {
          "kind": "advice",
          "text": "상체 각도가 스윙 중에 달라지지 않는지 확인해 보세요."
        }
      ],
      "stats": [
        {
          "label": "어드레스→탑 변화량",
          "at": "탑",
          "value_text": "+12°",
          "pro_text": "+3°",
          "range_text": null,
          "tail_pct": null
        }
      ],
      "sources": [
        "dynamics"
      ],
      "media": {
        "keyframe": "frames/0.jpg",
        "skeleton_compare": "skeletons/0.jpg",
        "keyframes": [
          "frames/0.jpg",
          "frames/3.jpg"
        ]
      }
    },
    {
      "rank": 3,
      "id": "weight_shift",
      "score": 52,
      "status": {
        "code": "adjust",
        "label": "조금 더 안정적으로"
      },
      "event": {
        "key": "Impact",
        "label": "임팩트"
      },
      "title": "임팩트 체중 이동 차이",
      "topic": "체중 이동 차이",
      "subtitle": "골반이 프로보다 옆으로 많이 이동해요(0.34배).",
      "body": [
        {
          "kind": "advice",
          "text": "몸이 옆으로 밀리는 것(스웨이)은 아닌지 확인해 보세요."
        }
      ],
      "stats": [
        {
          "label": "체중 이동",
          "at": "임팩트",
          "value_text": "0.34배",
          "pro_text": "0.23배",
          "range_text": "0.5~1.5",
          "tail_pct": 40.0
        }
      ],
      "sources": [
        "motion"
      ],
      "media": {
        "keyframe": "frames/5.jpg",
        "skeleton_compare": "skeletons/5.jpg",
        "keyframes": [
          "frames/5.jpg"
        ]
      }
    }
  ],
  "empty_state": null,
  "praise": [
    {
      "id": "tempo",
      "title": "스윙 리듬",
      "subtitle": "스윙 리듬이 프로 평균과 가까워요(백스윙:다운스윙 = 2.8:1, 프로 평균 3.9:1).",
      "score": 100,
      "event": null,
      "media": {
        "keyframe": null,
        "skeleton_compare": null,
        "keyframes": []
      }
    },
    {
      "id": "shoulder_turn_top",
      "title": "탑 어깨 회전",
      "subtitle": "백스윙에서 어깨 회전이 프로 범위 안이에요(1.0).",
      "score": 97,
      "event": {
        "key": "Top",
        "label": "탑"
      },
      "media": {
        "keyframe": "frames/3.jpg",
        "skeleton_compare": "skeletons/3.jpg",
        "keyframes": [
          "frames/3.jpg"
        ]
      }
    }
  ],
  "events": [
    {
      "key": "Address",
      "label": "어드레스",
      "frame_index": 10,
      "issue_ids": [],
      "media": {
        "keyframe": "frames/0.jpg",
        "skeleton_compare": "skeletons/0.jpg",
        "keyframes": [
          "frames/0.jpg"
        ]
      }
    },
    {
      "key": "Impact",
      "label": "임팩트",
      "frame_index": 60,
      "issue_ids": [
        "elbow",
        "weight_shift"
      ],
      "media": {
        "keyframe": "frames/5.jpg",
        "skeleton_compare": "skeletons/5.jpg",
        "keyframes": [
          "frames/5.jpg"
        ]
      }
    }
  ],
  "analysis_notes": [
    "정면 영상 기준의 2D 추정이라 회전각은 근사값이에요."
  ]
}
```
