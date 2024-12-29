#  Progressive Dependency Representation Learning for Stock Ranking in Uncertain Risk Contrasting
## Requirements
We implement PDU and other methods through the following dependencies:
- torch                            2.1.0
- numpy                         1.24.3
- scikit-learn                  1.3.0
- torch-cluster	         1.6.3+pt21cu121	
- torch-geometric	   2.4.0	
- torch-scatter        	 2.1.2+pt21cu121	
- torch-sparse	         0.6.18+pt21cu121	
- torchaudio	            2.1.0	
- torchvision	           0.16.0	
- tqdm	                     4.66.1	
## Dataset
Access channels for each of the four data sets can be obtained through the following four links:

[NASDAQ](https://www.nasdaq.com/news-and-insights)

[ NYSE](https://cn.investing.com/indices/nyse-composite-historical-data)

[SNP500](https://github.com/dmis-lab/hats)

[SZSE](https://www.szse.cn/market/product/stock/list/index.html)

## Usage
Before running the code, ensure the package structure of PDU is as follows:

```
.
├── Train
│   ├── evaluator
│   ├── load_data
│   ├── PDU
│   ├── PDUArch
│   ├── utils
│   └── main
└── data
```


### Train
```bash
python main.py
```
