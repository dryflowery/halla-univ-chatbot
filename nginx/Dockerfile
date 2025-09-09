# nginx/Dockerfile

FROM nginx:alpine

RUN rm /etc/nginx/conf.d/default.conf

COPY nginx/chatbot.conf /etc/nginx/conf.d/chatbot.conf
COPY frontend/ /usr/share/nginx/html

EXPOSE 80 443
