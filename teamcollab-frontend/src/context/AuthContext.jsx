import { createContext, useContext, useState, useEffect } from "react";
import { jwtDecode } from "jwt-decode";
import { useNavigate } from "react-router-dom";
import { toast } from "react-toastify";

const AuthContext = createContext();

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
  const [loading, setLoading] = useState(true);
  let [authToken, setAuthToken] = useState(() =>
    localStorage.getItem("authToken")
      ? JSON.parse(localStorage.getItem("authToken"))
      : null
  );
  let [user, setUser] = useState(() =>
    localStorage.getItem("authToken")
      ? jwtDecode(localStorage.getItem("authToken"))
      : null
  );

  const navigate = useNavigate();

  const isTokenExpired = (token) => {
    try {
      const decoded = jwtDecode(token);
      const currentTime = Date.now() / 1000; // in seconds
      return decoded.exp < currentTime;
    } catch (error) {
      return true; // If decoding fails, treat as expired
    }
  };

  useEffect(() => {
    if (authToken) {
      const expired = isTokenExpired(authToken.access);
      if (expired) {
        logoutUser(); // force logout
        return;
      }
    }

    let interval = setInterval(() => {
      if (authToken) {
        const expired = isTokenExpired(authToken.access);
        if (expired) {
          logoutUser();
        } else {
          updateToken();
        }
      }
    }, 1000 * 10 * 60); // check every 10 seconds, or adjust as needed

    return () => clearInterval(interval);
    setLoading(false);
  }, [authToken]);

  let login = async (e) => {
    e.preventDefault();

    let response = await fetch(
      "api/token/", // local test ...
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username: e.target.username.value.toLowerCase(),
          password: e.target.password.value,
        }),
      }
    );

    if (response.status === 500) toast.error(`Server Error !`); // check vite config

    let data = await response.json();

    if (response.status === 200) {
      setAuthToken(data);
      setUser(jwtDecode(data.access));
      localStorage.setItem(
        "authToken",
        JSON.stringify({
          refresh: data.refresh,
          access: data.access,
        })
      );

      toast.success("Welcome Back!");
      navigate("/");
    } else {
      toast.error(`Error Logging in.. ` + data.detail);
    }
  };

  const logoutUser = () => {
    setUser(null);
    localStorage.removeItem("authToken");
    toast({
      title: "Logged Out",
      description: "You have been successfully logged out.",
    });
    navigate("/");
  };

  return (
    <AuthContext.Provider value={{ user, login, logoutUser, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;
