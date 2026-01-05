FROM node:20-alpine
WORKDIR /app
COPY services/frontend/package.json services/frontend/package-lock.json* /app/
RUN npm install
COPY services/frontend /app
EXPOSE 5173
CMD ["npm", "run", "dev"]
