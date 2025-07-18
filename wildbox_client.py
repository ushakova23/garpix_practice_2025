# wildbox_client.py
import os
import requests
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


AUTH_TOKEN = os.getenv("AUTH_TOKEN")
COMPANY_ID = os.getenv("COMPANY_ID")
USER_ID = os.getenv("USER_ID")
COOKIE_STRING = os.getenv("COOKIE_STRING")

if not all([AUTH_TOKEN, COMPANY_ID, USER_ID, COOKIE_STRING]):
    raise ValueError("Необходимо задать все переменные окружения в .env файле (AUTH_TOKEN, COMPANY_ID, USER_ID, COOKIE_STRING)")

HEADERS = {
    'Authorization': AUTH_TOKEN,
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
    'CompanyID': COMPANY_ID,
    'UserID': USER_ID,
    'Referer': 'https://wildbox.ru/dashboard/search-tops-analysis/formed',
    'Sec-Fetch-Dest': 'empty', 'Sec-Fetch-Mode': 'cors', 'Sec-Fetch-Site': 'same-origin',
    'Time-Zone': 'Europe/Moscow',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36'
}

# Преобразуем строку cookie в словарь
COOKIES = {cookie.split('=')[0]: cookie.split('=')[1] for cookie in COOKIE_STRING.split('; ')}

DATE_TO = datetime.now().strftime('%Y-%m-%d')
DATE_FROM = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')


def get_top_products(city: str, search_query: str, total_limit: int) -> list:
    """Получает топ товаров с пагинацией."""
    url = "https://wildbox.ru/api/wb_dynamic/products/"
    base_params = {'city': city, 'wb_search': search_query, 'period': 30, 'ordering': 'product_ids', 'extra_fields': 'id,brand,auto_adv,cpm,position_dynamic,wh_avg_position,expected_position'}
    all_results = []
    page_size = 100
    offset = 0

    while len(all_results) < total_limit:
        params = base_params.copy()
        params['limit'] = page_size
        params['offset'] = offset
        try:
            response = requests.get(url, headers=HEADERS, cookies=COOKIES, params=params, timeout=45)
            response.raise_for_status()
            results = response.json().get('results', [])
            if not results: break
            all_results.extend(results)
            offset += page_size
            time.sleep(0.5)
        except requests.exceptions.RequestException as e:
            print(f"  -> ОШИБКА при запросе топа (offset={offset}) для '{city}': {e}")
            break
    return all_results[:total_limit]

def get_product_details(product_id: int) -> dict:
    """Получает детальную информацию по одному товару."""
    url = "https://wildbox.ru/api/wb_dynamic/products/"
    extra_fields = ('orders,proceeds,in_stock_percent,lost_proceeds,orders_dynamic,proceeds_dynamic,quantity,price,discount,weighted_price,rating,seller,brand,subject,images,promos,sales_speed,old_price')
    params = {'product_ids': product_id, 'date_from': DATE_FROM, 'date_to': DATE_TO, 'extra_fields': extra_fields}
    try:
        response = requests.get(url, headers=HEADERS, cookies=COOKIES, params=params, timeout=30)
        response.raise_for_status()
        results = response.json().get('results', [])
        return results[0] if results else {}
    except requests.exceptions.RequestException: return {}

def get_brand_details(brand_id: int) -> dict:
    """Получает информацию по бренду."""
    url = "https://wildbox.ru/api/wb_dynamic/brands/"
    params = {'brand_ids': brand_id, 'date_from': DATE_FROM, 'date_to': DATE_TO, 'extra_fields': 'rating,reviews'}
    try:
        response = requests.get(url, headers=HEADERS, cookies=COOKIES, params=params, timeout=30)
        response.raise_for_status()
        results = response.json().get('results', [])
        return results[0] if results else {}
    except requests.exceptions.RequestException: return {}
    
def get_warehouse_positions(product_id, search_query):
    """
    Получает позиции товара по городам по конкретному запросу.
    """
    import urllib.parse
    
    print(f"[API Client] Запрос позиций по складам для товара ID: {product_id}")
    url = "https://wildbox.ru/api/monitoring/positions/"
    
    encoded_phrase = urllib.parse.quote(search_query, safe='')
    
    params = {'product_id': product_id, 'phrase': search_query, 'pages_max': 30}
    
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        'Authorization': AUTH_TOKEN,  
        'CompanyID': COMPANY_ID,
        'Connection': 'keep-alive',
        'Referer': f'https://wildbox.ru/dashboard/position/formed?product_id={product_id}&phrase={encoded_phrase}',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'Time-Zone': 'Europe/Moscow',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
        'UserID': USER_ID,
        'sec-ch-ua': '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"'
    }
    
    try:
        encoded_params = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        full_url = f"{url}?{encoded_params}"
        
        print(f"[API Client] Полный URL: {full_url}")
        print(f"[API Client] Заголовки: {headers}")
        
        response = requests.get(full_url, headers=headers, cookies=COOKIES, timeout=45)
        
        print(f"[API Client] Статус ответа: {response.status_code}")
        print(f"[API Client] Заголовки ответа: {dict(response.headers)}")
        print(f"[API Client] Текст ответа (первые 500 символов): {response.text[:500]}")
        
        if response.status_code == 404:
            print("  -> Внимание: Эндпоинт позиций вернул ошибку 404 (Not Found).")
            return []
        
        if response.status_code == 403:
            print("  -> Ошибка авторизации 403. Проверьте токен и права доступа.")
            return []
            
        response.raise_for_status()
        
        try:
            data = response.json()
        except ValueError as e:
            print(f"  -> Ошибка парсинга JSON: {e}")
            print(f"  -> Полный текст ответа: {response.text}")
            return []
        
        print(f"[API Client] Тип данных: {type(data)}")
        print(f"[API Client] Получено записей: {len(data) if isinstance(data, list) else 'не список'}")
        print(f"[API Client] Данные: {data}")
        
        # API может вернуть словарь с ошибкой вместо списка
        if isinstance(data, dict) and data.get('detail'):
            print(f"  -> API вернул деталь: {data['detail']}")
            return []
            
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"  -> Ошибка при запросе позиций по складам {product_id}: {e}")
        return []
