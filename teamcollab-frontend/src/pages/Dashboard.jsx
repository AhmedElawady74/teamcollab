import { useContext, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import { useTask } from "@/context/TaskContext";
import Navbar from "@/components/Navbar";
import SideMenu from "@/components/SideMenu";
import DashboardStats from "@/components/DashboardStats";
import TaskCard from "@/components/TaskCard";
import Notifications from "@/components/Notifications";
import AuthContext from "../context/AuthContext";

const Dashboard = () => {
  const {user} = useContext(AuthContext)
  const { getUserTasks, loading: taskLoading } = useTask();
  const navigate = useNavigate();

  useEffect(() => {
    if (!user) {
      navigate("/login");
    }
  }, [user, navigate]);

  if (taskLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  if (!user) return null;

  const userTasks = getUserTasks(user.id);
  const recentTasks = userTasks.slice(0, 3);

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <Navbar />
      <div className="flex flex-1">
        <div className="hidden md:block">
          <SideMenu />
        </div>
        <div className="flex-1 p-6">
          <div className="mb-6">
            <h1 className="text-2xl font-bold">Welcome, {user.username}</h1>
            <p className="text-gray-600">
              Here's what's happening with your tasks
            </p>
          </div>

          <div className="mb-8">
            <DashboardStats />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <div className="mb-4 flex justify-between items-center">
                <h2 className="text-xl font-semibold">Your Recent Tasks</h2>
                <button
                  onClick={() => navigate("/tasks")}
                  className="text-primary hover:text-blue-700 text-sm"
                >
                  View All
                </button>
              </div>
              <div>
                {recentTasks.length > 0 ? (
                  recentTasks.map((task) => (
                    <TaskCard key={task.id} task={task} />
                  ))
                ) : (
                  <div className="text-center py-8 bg-white rounded-lg border">
                    <p className="text-gray-500">
                      No tasks assigned to you yet
                    </p>
                  </div>
                )}
              </div>
            </div>
            <div>
              <Notifications />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
