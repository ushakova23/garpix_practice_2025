## Описание проекта 
Проект представляет собой реализацию API с возможностью интегрирования, способного анализировать факторы ранжирования товаров на маркетплейсе Wildberries и предлагать рекомендации по изменению факторов для поднятия товара в топе и увеличению выручки, принимая на вход местоположение пользователя, запрос и артикул товара.

## Пример работы приложения
<img width="1536" height="279" alt="image" src="https://github.com/user-attachments/assets/896116f8-bafc-418f-908c-b6a411843ebf" />
<img width="1564" height="762" alt="image" src="https://github.com/user-attachments/assets/390a302d-ae2b-4563-9f24-18aeae159e43" />
<img width="1566" height="719" alt="image" src="https://github.com/user-attachments/assets/89877a77-f98c-4eae-8155-a28f3f7d76ee" />


## Инструкция по установке через Docker
1. Клонируйте репозиторий:
```
git clone https://github.com/ushakova23/garpix_practice_2025.git
cd garpix_practice_2025
```
2. В корне проекта создайте файл .env и добавьте в него переменные окружения:
```
AUTH_TOKEN=ваш_токен
COMPANY_ID=ваш_company_id
USER_ID=ваш_user_id
COOKIE_STRING=ваши_cookies
```
 Замените ваш_токен, ваш_company_id, ваш_user_id и ваши_cookies на реальные значения.

3. Соберите и запустите контейнер:
```
docker-compose up -d --build
```
4. API будет доступно по адресу:
```
http://localhost:8000
```
