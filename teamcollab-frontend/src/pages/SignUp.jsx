import { useState } from "react";
import axios from "axios";
import { toast } from "react-toastify";
import { Link, useNavigate } from "react-router-dom";

const SignUp = () => {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    password2: "",
  });

  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (formData.password !== formData.password2) {
      toast.error("Passwords do not match");
      return;
    }

    try {
      const response = await axios.post("http://localhost:8000/api/signup/", {
        ...formData,
      });
      toast.success(response.data.message);
      navigate("/login"); // Redirect to login page
    } catch (error) {
      console.error("Signup failed:", error.response?.data);
      toast.error("Signup failed. Please check your details.");
    }
  };
  return (
    <div className="container my-2 text-center">
      <div className="col-md-10 mx-auto col-lg-5">
        <img
          className="mb-2"
          src="src/assets/logo.png"
          alt=""
          width={120}
          height={87}
        />
        <h1 className="h3 mb-3 fw-normal">Create New Patient Account</h1>

        <form
          onSubmit={handleSubmit}
          className="p-4 p-md-5 border rounded-3 bg-light"
        >
          <div className="form-floating mb-3">
            <input
              type="text"
              name="username"
              className="form-control"
              id="floatingInput"
              placeholder=""
              onChange={handleChange}
            />
            <label htmlFor="floatingInput">Username</label>
          </div>
          <div className="form-floating my-2">
            <input
              type="email"
              name="email"
              className="form-control"
              id="floatingInput2"
              placeholder="name@example.com"
              onChange={handleChange}
            />
            <label htmlFor="floatingInput2">Email address</label>
          </div>
          <div className="form-floating mb-3">
            <input
              type="password"
              name="password"
              className="form-control"
              id="floatingPassword"
              placeholder="Password"
              onChange={handleChange}
            />
            <label htmlFor="floatingPassword">Password</label>
          </div>
          <div className="form-floating mb-3">
            <input
              type="password"
              name="password2"
              className="form-control"
              id="floatingPassword2"
              placeholder="Repeat Password"
              onChange={handleChange}
            />
            <label htmlFor="floatingPassword2">Repeat Password</label>
          </div>
          <button className="w-100 btn btn-lg btn-success my-2" type="submit">
            Sign up
          </button>
          <hr className="my-4" />
          <small className="text-muted">
            By clicking Sign up, you agree to the terms of use.
          </small>

          <Link
            to={"/login"}
            className="w-100 btn btn-lg btn-outline-primary my-2"
          >
            Already Have Account, Login Here!
          </Link>
        </form>
      </div>
    </div>
  );
};

export default SignUp;
