from algomodels import D1, D5

if __name__ == '__main__':
    # запускает общую дефолтную модель для указанных тикеров. Но папка algomodels содержит модель под каждый тикер
    # отдельно, которые нужно будет в дальнейшем развивать
    tickers = ['ALRS', 'GAZP', 'LKOH', 'MGNT', 'NVTK', 'PHOR', 'PLZL', 'POLY', 'ROSN', 'SBER', 'SNGS', 'TATN',
               'TCSG', 'YNDX']

    for name in tickers:
        D1.train(name)
        D5.train(name)