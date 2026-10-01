# data

원본 데이터는 용량(약 7.8GB) 때문에 저장소에 포함하지 않는다.

## 받는 곳

- Kaggle : [MIT-Stanford Dataset](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle) (로그인 필요)
- 원 출처 : [data.matr.io](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) (CC BY 4.0)

| 배치 | 파일 | 크기 (bytes) |
|---|---|---|
| Batch 1 (학습) | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | 3,025,320,241 |
| Batch 2 (테스트) | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | 2,022,599,329 |
| Batch 3 (추가 테스트) | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | 3,236,690,412 |
| extra (사용 안 함) | `2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat` | 89,125,795 |

## 배치 방법

```
data/
├── raw/          # 위 .mat 파일을 그대로 둔다
└── processed/    # python src/preprocess.py 실행 시 생성 (b1.pkl, b2.pkl, b3.pkl)
```

`src/preprocess.py` 는 셀마다 수명·충전 정책·사이클 요약값 전체와 앞쪽 120 사이클의 `Qdlin`/`Tdlin` 곡선만 뽑아 pickle 로 저장한다.
