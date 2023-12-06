import json

result = {
  "1": {
    "tickers": [],
    "prices": [],
    "predict_profit": []
  },
  "2": {
    "tickers": [
    ],
    "prices": [
    ],
    "predict_profit": [
    ]
  },
  "3": {
    "tickers": [],
    "prices": [],
    "predict_profit": []
  }
}

def read_json():
    json_name = "tickers_data.json"
    with open(json_name, "r+") as file:
        data = json.load(file)

    return data

def create_leaderboard(data):
    json_name = "leaderboard.json"
    with open(json_name, "w") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def get_top_n_for_leaderboards(data, n):
    for key, leaderboard in data.items():
        for i in range(n):
            # получаем индекс максимального элемента
            max_index = leaderboard['predict_profit'].index(max(leaderboard['predict_profit']))
            result[key]['tickers'].append(leaderboard['tickers'].pop(max_index))
            result[key]['prices'].append(leaderboard['prices'].pop(max_index))
            result[key]['predict_profit'].append(leaderboard['predict_profit'].pop(max_index))

    return result

data = read_json()
leaderboard_data = get_top_n_for_leaderboards(data, 2)
create_leaderboard(leaderboard_data)

count = 0

while True:
    try:
        url = 'http://' + respos_url + '/api/v1/task/complied'
        response = requests.post(url, json=result)
        if response.status_code == 200:
            print("Запрос успешно отправлен:")
            break
    except Exception as err:
        print("Ошибка отправка запроса на API:", err)

    # Делаем повторные попытки в случае ошибки
    if count >= 5:
        break

    count += 1

def send_result
