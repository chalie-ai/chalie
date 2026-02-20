FROM nginx:1.25-alpine

RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY interface /usr/share/nginx/html
COPY brain /usr/share/nginx/html/brain
COPY on-boarding /usr/share/nginx/html/on-boarding

RUN chmod -R 755 /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
