import { useContext, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import { useTask } from "@/context/TaskContext";
import Navbar from "@/components/Navbar";
import SideMenu from "@/components/SideMenu";
import TaskCard from "@/components/TaskCard";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Search, Filter } from "lucide-react";
import { TASK_STATUS, TASK_PRIORITY } from "@/lib/constants";
import AuthContext from "../context/AuthContext";

const TasksList = () => {
  // const { user, loading: authLoading } = useAuth();
  const { user } = useContext(AuthContext);
  const { getUserTasks, loading: taskLoading } = useTask();
  const navigate = useNavigate();

  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [priorityFilter, setPriorityFilter] = useState("all");
  const [filteredTasks, setFilteredTasks] = useState([]);

  useEffect(() => {
    if (!user) {
      navigate("/login");
    }
  }, [user, navigate]);

  useEffect(() => {
    if (user) {
      const userTasks = getUserTasks(user.user_id);
      // Apply filters
      let filtered = [...userTasks];

      // Filter by search term
      if (searchTerm) {
        filtered = filtered.filter(
          (task) =>
            task.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
            task.description.toLowerCase().includes(searchTerm.toLowerCase())
        );
      }

      // Filter by status
      if (statusFilter !== "all") {
        filtered = filtered.filter((task) => task.status === statusFilter);
      }

      // Filter by priority
      if (priorityFilter !== "all") {
        filtered = filtered.filter((task) => task.priority === priorityFilter);
      }

      setFilteredTasks(filtered);
    }
  }, [user, getUserTasks, searchTerm, statusFilter, priorityFilter]);

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

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <Navbar />
      <div className="flex flex-1">
        <div className="hidden md:block">
          <SideMenu />
        </div>
        <div className="flex-1 p-6">
          <div className="mb-6">
            <h1 className="text-2xl font-bold">My Tasks</h1>
            <p className="text-gray-600">View and manage your assigned tasks</p>
          </div>

          {/* Search and Filter */}
          <div className="mb-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="relative">
              <Search
                className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400"
                size={18}
              />
              <Input
                placeholder="Search tasks..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            <div className="flex items-center space-x-2">
              <Filter size={18} className="text-gray-500" />
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger className="flex-1">
                  <SelectValue placeholder="Filter by status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Statuses</SelectItem>
                  <SelectItem value={"pending"}>Pending</SelectItem>
                  <SelectItem value={"in_progress"}>In Progress</SelectItem>
                  <SelectItem value={"completed"}>Completed</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center space-x-2">
              <Filter size={18} className="text-gray-500" />
              <Select value={priorityFilter} onValueChange={setPriorityFilter}>
                <SelectTrigger className="flex-1">
                  <SelectValue placeholder="Filter by priority" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Priorities</SelectItem>
                  <SelectItem value={"high"}>High</SelectItem>
                  <SelectItem value={"medium"}>Medium</SelectItem>
                  <SelectItem value={"low"}>Low</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Task List */}
          <div>
            {filteredTasks.length > 0 ? (
              filteredTasks.map((task) => (
                <TaskCard key={task.id} task={task} />
              ))
            ) : (
              <div className="text-center py-16 bg-white rounded-lg border">
                <p className="text-gray-500 mb-2">No tasks found</p>
                <p className="text-sm text-gray-400">
                  Try adjusting your filters
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TasksList;
