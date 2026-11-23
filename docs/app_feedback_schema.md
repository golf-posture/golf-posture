# 앱용 피드백 JSON 스키마 (`feedback_app.json`, schema_version 3)

분석이 끝나면 `outputs/<이름>/feedback_app.json`이 만들어집니다. 앱의 "분석 결과 상세" 화면을 이 파일 하나로 그릴 수 있게 정리한 것입니다. 만드는 코드는 `pipeline/app_feedback.py`의 `build_app_payload`이고, 문구와 기준값은 같은 파일 위쪽 상수에서 바꿉니다.

기존 `feedback_ko.txt`(상세)와 `feedback_user.txt/json`(일반 사용자용 텍스트)은 이 파일과 별개로 그대로 만들어집니다. 앱 JSON을 만드는 데 실패해도 그 파일들은 영향을 받지 않습니다.

**v2에서 바뀐 점**: 점수 체계가 새로 바뀌었습니다(구간 선형 항목 점수, 이벤트별 점수, 이벤트 가중 자세 점수). 자세 점수는 0~10이 아니라 0~100이고, `score_inputs`는 없어졌습니다. 카드 `body[]`에는 `score`, `evidence`, `also` 종류가 추가되었고 카드에 `worst_item`, 이벤트에 `score`가 생겼습니다.

## 1. 화면과 필드 대응

| 화면 요소 | JSON 경로 |
|---|---|
| 자세 점수 (0~100) | `summary.posture_score` |
| 스윙 리듬 (예: 2.8 : 1) | `summary.tempo.text`, `summary.tempo.caption` |
| 분석 불확실 안내 | `notice` (없으면 `null`) |
| 자세 진단 탭: 문제점 카드 (최대 5개, 점수 낮은 순) | `issues[]` |
| 카드 제목, 부제 | `issues[].title`, `issues[].subtitle` |
| 카드 점수 막대와 상태 라벨 | `issues[].score`, `issues[].status` |
| 카드 하단 이미지 | `issues[].media.keyframe` (+ `skeleton_compare`, `keyframes`) |
| 카드를 눌렀을 때 자세한 피드백 | `issues[].body[]` |
| 문제 없음 화면 | `state == "no_issues"`, `empty_state` |
| 풀스윙 탭: 영상과 8개 이벤트 버튼 | `events[]` (`frame_index`로 이동) |
| 칭찬 탭 | `praise[]` |

## 2. 점수 체계

모든 점수는 0~100이고 같은 척도입니다. 높을수록 프로와 가깝습니다.

**항목 점수** (가장 작은 측정값)

프로 통계에서 나온 z(`|z|`)를 점수로 바꿉니다. 구간 선형이라 프로 범위 안은 완만하게, 밖은 가파르게 깎입니다.

| \|z\| | 0 | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 | 4 이상 |
|---|---|---|---|---|---|---|---|---|
| 점수 | 100 | 92.5 | 85 | 67.5 | 50 | 32.5 | 15 | 0 |

- 프로 범위 안(`|z|` 1 이하)은 85점 이상, "개선 필요"(`|z|` 2 초과)는 50점 미만입니다.
- 각도는 평균과 표준편차, 동작 지표와 스윙 다이내믹스는 중앙값과 보정 MAD로 z를 구합니다. 같은 점수라도 심각도가 완전히 같지는 않습니다.

**항목의 종류** (모든 항목이 점수 계산에 들어갑니다)

| 종류 | 개수 | 소속 이벤트 |
|---|---|---|
| 각도 (뒤쪽 팔꿈치, 타깃 쪽 팔꿈치, 어깨선, 골반선, 상체 기울기) | 5개 x 이벤트 8개 = 40개 | 그 이벤트 |
| 동작 지표 (어깨 회전, 어깨·골반 꼬임, 골반 회전, 체중 이동, 머리 위치, 골반 높이, 발 간격) | 7개 | 끝나는 이벤트 (탑 3개, 임팩트 4개) |
| 스윙 다이내믹스 (이벤트 사이 변화량, 5개 각도 x 7개 구간) | 35개 | 구간이 끝나는 이벤트 |

스윙 리듬(템포)은 이벤트에 속하지 않아서 점수에 들어가지 않고, 상단에 비율로만 보여줍니다.

**이벤트 점수** (`events[].score.items`)

그 이벤트에 속한 항목 점수의 가중 평균입니다. 항목의 가중치는 신뢰도에 따라 high 1.0, medium 0.6, low 0.2이고, 측정되지 않은 항목은 0점이 아니라 평균에서 빠집니다.

**이벤트 점수, 결합 점수 병행** (`events[].score.combined`)

현구님의 이벤트별 Mahalanobis 결합 점수(`deviation_percentile`)를 같은 척도로 바꾼 값입니다. 프로 분포에서 그만큼 특이하려면 필요한 `|z|`로 환산해서 항목 점수와 같은 구간 선형 점수로 바꿉니다. 각도 5개만 쓰는 결합 점수라 항목 기반 점수와 다를 수 있고, 비교용으로 병행합니다.

**자세 점수** (`summary.posture_score`)

이벤트 점수를 이벤트 가중치로 가중 평균한 값입니다. 결합 점수로 같은 계산을 한 값은 `summary.posture_score_alt.value`입니다.

| 어드레스 | 토업 | 백스윙 | 탑 | 다운스윙 | 임팩트 | 팔로스루 | 피니시 |
|---|---|---|---|---|---|---|---|
| 1.0 | 0.6 | 0.8 | 1.2 | 1.0 | 1.5 | 0.7 | 0.5 |

이 가중치는 임시값입니다. 어드레스, 탑, 임팩트가 핵심 구간이라는 일반적인 가정으로 정했고, 코치 확인이 필요합니다. 값은 `EVENT_WEIGHTS`에서 바꿉니다.

**카드 점수** (`issues[].score`)

카드를 이루는 항목 중 가장 나쁜 항목의 점수입니다. 이벤트 점수와는 묶는 방향이 다릅니다. 카드는 "무엇을 고쳐야 하는가"(같은 문제를 이벤트 방향으로 묶어 가장 심한 값), 이벤트 점수는 "어느 시점이 약한가"(같은 이벤트를 문제 방향으로 묶어 평균)를 말해줍니다. 상태 라벨은 50 미만 `priority`, 68 미만 `adjust`, 그 외 `recheck`이고 기준값은 임시입니다.

## 3. 화면 상태

| 조건 | `state` |
|---|---|
| 문제점 카드가 1~5개 | `issues` (개수는 `issues.length`) |
| 문제점 카드가 없음 | `no_issues` (`empty_state`에 문구가 들어옵니다) |

문제 후보가 모자라도 개수를 채우지 않습니다. 분석이 불확실하면 `reliability`가 `low`가 되고 `notice`에 안내 문구가 들어옵니다. `notice.code`는 `tempo_excluded`, `many_unavailable`, `mock_events` 중 하나입니다.

## 4. 필드 정의

**최상위**: `schema_version`, `meta{handedness, mock_events}`, `state`, `reliability`(`normal`/`low`), `notice{code, message}`/`null`, `empty_state{title, message}`/`null`, `analysis_notes[]`

**`summary`**

| 필드 | 설명 |
|---|---|
| `posture_score` | 0~100 정수. 이벤트 가중 평균. 계산할 항목이 없으면 `null` |
| `posture_score_basis` | `event_weighted_items` |
| `posture_score_alt` | `{basis: "combined_mahalanobis", value}` 결합 점수 기반 비교값 |
| `event_weights` | 이벤트별 가중치 |
| `analyzed_items` | 점수 계산에 쓴 항목 수 (화면에 보여줄 값은 아닙니다) |
| `tempo` | `{status, ratio, text, caption, pro_mean}`. `status`가 `excluded`이면 `ratio`는 `null`, `text`는 `"-"` |

**`issues[]`** (최대 5개, 점수가 낮은 카드부터)

| 필드 | 설명 |
|---|---|
| `rank`, `id` | 1부터 시작하는 순위, `elbow` / `spine` / `weight_shift` 등 |
| `score`, `status` | 0~100 점수, `{code, label}` ("먼저 확인해요", "조금 더 안정적으로", "한 번 더 확인해요") |
| `event` | `{key, label}` 카드의 대표 이벤트 |
| `title`, `topic` | 중립 제목 "이벤트 + 문제점 요약"(방향 표현 없음), 이벤트를 뺀 요약 |
| `subtitle` | 한 줄 피드백 요약 |
| `worst_item` | 카드에서 가장 나쁜 항목 `{kind, event, label, score}`. `kind`는 `angle`, `dynamics`, `motion` |
| `body[]` | 자세한 피드백. `{kind, text}` 종류는 아래 표 참고 |
| `stats[]` | 선택. 내 값과 프로 값 비교 |
| `sources[]` | 카드를 만든 분석: `angle`, `dynamics`, `motion` |
| `media` | `{keyframe, skeleton_compare, keyframes[]}` |

`body[]`의 종류와 순서 (없는 종류는 빠집니다. 같은 종류가 두 번 나올 수 있고, 앱이 모르는 종류는 일반 문장으로 보여주면 됩니다)

| `kind` | 내용 | 예시 |
|---|---|---|
| `score` | 점수가 나온 지점 | "임팩트에서 타깃 쪽 팔꿈치 각도 점수가 29점으로 가장 낮아요." |
| `evidence` | 점수의 근거(내 값과 프로 값) | "내 값은 100°, 프로 평균은 95°예요(프로는 보통 ±5° 이내)." |
| `direction` | 간단한 방향(팔꿈치 각도만) | "타깃 쪽 팔꿈치가 프로보다 더 펴져 있어요." |
| `also` | 같은 문제가 나타난 다른 곳 | "같은 문제가 임팩트 뒤쪽 팔꿈치(40점)에서도 보여요." |
| `change` | 이벤트 사이 변화량 | "어드레스에서 탑으로 가는 동안 …도 점수가 43점으로 낮아요." |
| `pattern` | 반복 여부 | "어드레스, 탑, 임팩트에서 반복돼요. 한 순간의 흔들림보다 스윙 흐름의 차이일 수 있어요." |
| `advice` | 어떻게 하면 되는지 (1~2개) | "이 구간은 팔이 빠르게 움직여요. 슬로모션으로 …" |
| `drill` | 연습 방법 (점수가 68 미만인 카드에만) | "연습: 빈 스윙을 천천히 하며 임팩트에서 멈추고, …" |
| `caution` | 오차 가능성 안내 | "이 구간은 동작이 빨라 분석 오차가 생기기 쉬워요." |

**조언 문장이 정해지는 방식**

- 문제가 나타난 이벤트: 준비(어드레스~백스윙), 스윙(탑~임팩트), 마무리(팔로스루~피니시)에 따라 볼 곳이 달라집니다.
- 반복 횟수: 세 구간 이상에서 반복되면 스윙 흐름의 차이로, 한 구간이면 그 순간의 타이밍으로 안내합니다.
- 변화의 크기: 이벤트 사이 변화량이 프로보다 크면 "변화가 커지는 순간을 확인", 작으면 "프로 영상과 나란히 비교"로 안내합니다. 크기만 비교하고 방향은 말하지 않습니다.
- 방향: 팔꿈치 각도만 "더 펴져 / 더 굽혀져"로 말합니다(각도가 크면 팔이 펴진 것). 상체 기울기는 방향을 말하지 않습니다.
- 점수: 68 미만(`priority`, `adjust`)인 카드에만 연습 방법이 붙습니다. 연습 방법은 일반적인 연습 방법이고, 코치 검증을 거친 것은 아닙니다.
- 스위치: `app_feedback.py`의 `USE_DIRECTION_HINTS`와 `USE_DRILLS`를 `False`로 바꾸면 방향 문구와 연습 방법이 나오지 않습니다.

**`praise[]`** (칭찬 탭, 강한 칭찬부터 최대 12개): `{id, title, subtitle, score, event, body[], media}`

- 문제점 카드와 같은 모양이라 같은 방식으로 그릴 수 있습니다. 문제점으로 나온 부위는 칭찬에 나오지 않습니다.
- 칭찬이 나오는 경우는 세 가지입니다. 팔꿈치 각도, 상체 기울기, 어깨선, 골반선 기울기가 평가된 모든 이벤트에서 프로 범위 안일 때(`stable_*`), 동작 지표가 기준 안일 때, 스윙 리듬이 프로 평균에 가까울 때(`tempo`)입니다.
- `body[]`는 아래 순서입니다. 없는 종류는 빠집니다.

| `kind` | 내용 | 예시 |
|---|---|---|
| `score` | 점수와 수준 | "어깨 회전 점수가 98점으로, 프로 중앙값과 거의 같은 수준이에요." |
| `evidence` | 근거(내 값과 프로 값) | "내 값은 57°, 프로는 대부분 47°~80°예요(중앙값 64°)." |
| `praise` | 칭찬 문장 | "백스윙에서 어깨 회전이 프로 범위 안이에요(57°)." |
| `meaning` | 이 결과가 왜 좋은지 | "백스윙에서 어깨를 충분히 돌리면 몸 전체를 써서 힘을 모을 준비가 된 거예요." |
| `detail` | 구간별 안정도, 리듬 해석 | "가장 안정적인 구간은 탑(100점)이고, 가장 낮은 구간도 임팩트에서 86점이에요." |
| `keep` | 항목별 유지 조언 | "촬영할 때마다 어깨가 같은 깊이로 돌아가는지 비교해 보면 이 감각을 유지하기 쉬워요." |
| `caution` | 값을 읽을 때 주의할 점 (회전각, 카메라 영향을 받는 기울기) | "회전각은 정면 2D 영상에서 …근사값이라 실제와 조금 다를 수 있어요." |

점수 수준 문구는 95점 이상 "프로 중앙값과 거의 같은 수준", 90점 이상 "프로 분포의 한가운데에 가까워요", 그 외 "프로 범위 안에서 안정적이에요"입니다. "프로보다 낫다"는 표현은 쓰지 않습니다.

**`praise_header`**: 칭찬 탭 맨 위에 보여줄 한 줄 `{title, message}`입니다. 예: "잘한 점 10가지를 찾았어요. 이 부분은 그대로 유지하고, '임팩트 팔꿈치 각도 차이'를 먼저 확인해 보세요." 문제점이 없으면 "이번 스윙에서는 교정할 항목이 발견되지 않았어요."로 끝나고, 칭찬이 하나도 없으면 `null`입니다.

**`events[]`** (풀스윙 탭, 항상 8개)

`{key, label, frame_index, score, issue_ids, media}`. `score`는 `{items, combined, weight, item_count}`이고 `items`와 `combined`는 0~100 또는 `null`입니다. 화면에서 이벤트 점수를 쓰지 않아도 됩니다.

## 5. 문제점 카드가 만들어지는 방식

- 후보는 각도(팔꿈치, 상체 기울기 부위별), 동작 지표 7개, 스윙 다이내믹스(팔꿈치와 상체 기울기에 합쳐짐)에서 모읍니다.
- 신뢰도가 낮은 항목은 카드가 되지 않습니다. 어깨선과 골반선 기울기는 카메라 영향이 커서 카드로 올리지 않습니다(점수에는 들어갑니다).
- 카드 문장은 방향 표현 없이 점수가 나온 지점, 근거, 조언 순서로 씁니다. 방향("들림", "펴짐")이 실제 동작과 맞는지 코치 확인이 끝나지 않았기 때문입니다.

## 6. 백엔드 참고

- `media.*` 값은 분석을 돌린 서버의 파일 경로입니다. 앱에서 보려면 업로드하거나 URL로 바꿔서 내려주세요.
- 이벤트 시간은 `frame_index / fps`로 계산합니다. 분석 결과에 fps가 없어서 영상 메타데이터의 fps를 사용하세요.
- 값이 없는 필드는 키를 빼지 않고 `null`, 빈 배열로 내려갑니다.
- 필드를 추가하는 것은 `schema_version`을 올리지 않습니다. 앱은 모르는 필드를 무시하면 됩니다.
- 화면이 "6.8 / 10"이면 앱에서 `posture_score`를 10으로 나눠 보여주세요.

## 7. 보류와 한계

- 전문가용 / 일반용 탭 분리와 "프로 범위 안에 N개" 문장은 보류했습니다.
- 이벤트 가중치, 상태 라벨 기준(50, 68), 신뢰도 가중치(1.0, 0.6, 0.2)는 임시값입니다. 실제 영상으로 보정합니다.
- 연습 방법과 방향 문구는 코치 검증 전의 일반적인 안내입니다.
- 결합 점수는 병행 계산이라 자세 점수의 기준은 항목 기반 점수입니다. 두 값이 크게 다르면 원인을 확인합니다.
- 이벤트 위치 오차가 큰 영상(특히 다운스윙~임팩트)은 팔 각도가 흔들릴 수 있어 `caution`이 붙습니다.
- 정면 영상 기준의 2D 추정입니다.

## 8. 예시 (문제 3개, 칭찬과 이벤트는 일부만 표시)

```json
{
  "schema_version": 3,
  "meta": {
    "handedness": "right",
    "mock_events": false
  },
  "state": "issues",
  "reliability": "normal",
  "notice": null,
  "summary": {
    "posture_score": 91,
    "posture_score_basis": "event_weighted_items",
    "posture_score_alt": {
      "basis": "combined_mahalanobis",
      "value": null
    },
    "event_weights": {
      "Address": 1.0,
      "Toe-up": 0.6,
      "Mid-backswing": 0.8,
      "Top": 1.2,
      "Mid-downswing": 1.0,
      "Impact": 1.5,
      "Mid-follow-through": 0.7,
      "Finish": 0.5
    },
    "analyzed_items": 48,
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
      "score": 29,
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
          "kind": "score",
          "text": "임팩트에서 타깃 쪽 팔꿈치 각도 점수가 29점으로 가장 낮아요."
        },
        {
          "kind": "evidence",
          "text": "내 값은 100°, 프로 평균은 95°예요(프로는 보통 ±5° 이내)."
        },
        {
          "kind": "direction",
          "text": "타깃 쪽 팔꿈치가 프로보다 더 펴져 있어요."
        },
        {
          "kind": "also",
          "text": "같은 문제가 임팩트 뒤쪽 팔꿈치(40점), 다운스윙 타깃 쪽 팔꿈치(46점)에서도 보여요."
        },
        {
          "kind": "advice",
          "text": "이 구간은 팔이 빠르게 움직여요. 슬로모션으로 팔이 펴지고 굽어지는 타이밍을 프로 영상과 나란히 비교해 보세요."
        },
        {
          "kind": "advice",
          "text": "팔에 힘이 들어가 너무 곧게 펴져 있지 않은지 확인해 보세요."
        },
        {
          "kind": "drill",
          "text": "연습: 빈 스윙을 천천히 하며 임팩트에서 멈추고, 팔꿈치에 약간 여유가 있는 모양인지 거울로 확인해 보세요."
        },
        {
          "kind": "caution",
          "text": "이 구간은 동작이 빨라 분석 오차가 생기기 쉬워요. 다른 각도에서 한 번 더 촬영해 비교해 보면 더 정확해요."
        }
      ],
      "worst_item": {
        "kind": "angle",
        "event": {
          "key": "Impact",
          "label": "임팩트"
        },
        "label": "타깃 쪽 팔꿈치 각도",
        "score": 29
      },
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
      "score": 36,
      "status": {
        "code": "priority",
        "label": "먼저 확인해요"
      },
      "event": {
        "key": "Finish",
        "label": "피니시"
      },
      "title": "피니시 상체 기울기 차이",
      "topic": "상체 기울기 차이",
      "subtitle": "팔로스루에서 피니시로 가는 동안 상체 기울기 변화가 프로와 달라요.",
      "body": [
        {
          "kind": "score",
          "text": "팔로스루에서 피니시로 가는 동안 상체 기울기 변화 점수가 36점이에요."
        },
        {
          "kind": "evidence",
          "text": "프로는 이 구간에서 보통 14° 정도 변하는데, 내 스윙은 거의 변하지 않았어요."
        },
        {
          "kind": "advice",
          "text": "이 구간에서 프로는 상체 기울기가 변하는데 내 스윙은 변화가 작아요. 프로 영상과 이 구간을 나란히 놓고 어떻게 움직이는지 비교해 보세요."
        },
        {
          "kind": "drill",
          "text": "연습: 프로 영상의 팔로스루~피니시 구간을 반복해서 보고, 같은 속도로 천천히 따라 해 보세요."
        }
      ],
      "worst_item": {
        "kind": "dynamics",
        "event": {
          "key": "Finish",
          "label": "피니시"
        },
        "label": "상체 기울기 변화",
        "score": 36
      },
      "sources": [
        "dynamics"
      ],
      "media": {
        "keyframe": "frames/6.jpg",
        "skeleton_compare": "skeletons/6.jpg",
        "keyframes": [
          "frames/6.jpg",
          "frames/7.jpg"
        ]
      }
    },
    {
      "rank": 3,
      "id": "weight_shift",
      "score": 64,
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
          "kind": "score",
          "text": "체중 이동 점수가 64점이에요."
        },
        {
          "kind": "evidence",
          "text": "내 값은 0.34배, 프로는 대부분 0.5~1.5예요(중앙값 0.23배)."
        },
        {
          "kind": "advice",
          "text": "몸이 옆으로 밀리는 것(스웨이)은 아닌지 확인해 보세요."
        },
        {
          "kind": "drill",
          "text": "연습: 오른쪽 골반 바깥에 벽이나 의자를 가볍게 대고, 백스윙 동안 골반이 닿은 곳에서 떨어지지 않게 돌려 보세요."
        }
      ],
      "worst_item": {
        "kind": "motion",
        "event": {
          "key": "Impact",
          "label": "임팩트"
        },
        "label": "체중 이동",
        "score": 64
      },
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
      "score": 98,
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
      "key": "Impact",
      "label": "임팩트",
      "frame_index": 60,
      "score": {
        "items": 80,
        "combined": null,
        "weight": 1.5,
        "item_count": 9
      },
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
