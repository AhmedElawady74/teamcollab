import axios from "axios";

const baseURL = "http://localhost:8000/api/"; // test localhost environment ...
// const baseURL = "https://xxx.xxx.xxxx/api/"; // on railway.app testing...
const AxiosInstance = axios.create({
  baseURL: baseURL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
    accept: "application/json",
  },
});


AxiosInstance.interceptors.request.use(
  (config) => {
    const tokenString = localStorage.getItem("authToken");
    if (tokenString) {
      const tokenData = JSON.parse(tokenString); // 👈 Parse the string
      config.headers["Authorization"] = `Bearer ${tokenData.access}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);


export default AxiosInstance;
