import { createContext, useContext, useState, useEffect } from "react";
import { useToast } from "@/components/ui/use-toast";
import AuthContext, { useAuth } from "./AuthContext";
import AxiosInstance from "../components/utils/AxiosInstance";
import { toast } from "react-toastify";

const TaskContext = createContext();

export const useTask = () => useContext(TaskContext);

export const TaskProvider = ({ children }) => {
  const [tasks, setTasks] = useState([]);
  const [comments, setComments] = useState([]);
  const [notifications, setNotifications] = useState([]);
  const [loading, setLoading] = useState(true);
  const { user } = useContext(AuthContext);

  const getTasks = async () => {
    try {
      setLoading(true);
      const response = await AxiosInstance.get("tasks/");
      setTasks(response.data); // Set tasks directly
    } catch (error) {
      console.error("Error fetching tasks:", error);
      // Fallback to mock data if API fails
      await loadMockData();
    } finally {
      setLoading(false);
    }
  };

  const getComments = async () => {
    let response = await AxiosInstance.get(`comments/`);
    setComments(response.data);
  };

  // Load initial mock data
  useEffect(() => {
    if (user) {
      getTasks();
      getComments();
    }
    // Simulate API call to fetch tasks
    const fetchInitialData = async () => {
      try {
        // In a real app, these would be separate API calls
        await loadMockData();
        setLoading(false);
      } catch (error) {
        console.error("Error fetching initial data:", error);
        setLoading(false);
      }
    };
    fetchInitialData();
  }, [user]);

  // Load mock data
  const loadMockData = async () => {
    // Simulate API delay
    await new Promise((resolve) => setTimeout(resolve, 800));

    // Mock notifications
    const mockNotifications = [
      {
        id: 1,
        userId: user?.user_id || 1,
        text: 'You have been assigned a new task: "Implement User Authentication"',
        isRead: false,
        created_at: "2023-11-01T10:05:00Z",
      },
      {
        id: 2,
        userId: user?.user_id || 1,
        text: "John manager mentioned you in a comment",
        isRead: true,
        created_at: "2023-11-03T15:46:00Z",
      },
      {
        id: 3,
        userId: user?.user_id || 1,
        text: 'Task "Design Task Management UI" has been marked as completed',
        isRead: false,
        created_at: "2023-11-05T16:35:00Z",
      },
    ];

    // setTasks(mockTasks);
    // setComments(mockComments);
    setNotifications(mockNotifications);
  };

  // Add a new task
  const addTask = async (taskData) => {
    setLoading(true);
    // API call
    await new Promise((resolve) => setTimeout(resolve, 500));
    try {
      const res = await AxiosInstance.post(`tasks/`, taskData);
      if (res.status === 201 || res.status === 200) {
        toast.success("New Task Addedd Successfully!");
      } else {
        toast.error("Failed to add task.");
      }
    } catch (e) {
      toast.error(e.message);
    } finally {
      setLoading(false);
    }

    // Create notification for assigned user
    // if (taskData.assigned_to) {
    //   addNotification({
    //     userId: taskData.assigned_to.id,
    //     text: `You have been assigned a new task: "${taskData.title}"`,
    //   });
    // }
  };

  // Update a task
  const updateTask = async (taskId, updatedData) => {
    try {
      setLoading(true);
      try {
        const res = await AxiosInstance.put(`tasks/${taskId}/`, updatedData);
        if (res.status === 201 || res.status === 200) {
          toast.success("Task Updated Successfully!");
        } else {
          toast.error("Failed to add task.");
        }
      } catch (e) {
        toast.error(e.message);
      } finally {
        setLoading(false);
      }

      return

      // Create notifications for status changes
      if (updatedData.status && updatedData.status !== oldTask.status) {
        addNotification({
          userId: oldTask.assigned_to.id,
          text: `Task "${oldTask.title}" has been updated to ${updatedData.status}`,
        });
      }

      // Create notifications for reassignment
      if (
        updatedData.assigned_to &&
        updatedData.assigned_to.id !== oldTask.assigned_to.id
      ) {
        addNotification({
          userId: updatedData.assigned_to.id,
          text: `You have been assigned to the task: "${oldTask.title}"`,
        });
      }

      toast({
        title: "Task Updated",
        description: "The task has been successfully updated.",
      });

      return updatedTask;
    } catch (error) {
      console.error("Error updating task:", error);
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to update task. Please try again.",
      });
      return null;
    } finally {
      setLoading(false);
    }
  };

  // Delete a task
  const deleteTask = async (taskId) => {
    try {
      setLoading(true);

      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 500));

      const taskToDelete = tasks.find((task) => task.id === taskId);

      if (!taskToDelete) {
        throw new Error("Task not found");
      }

      setTasks((prevTasks) => prevTasks.filter((task) => task.id !== taskId));

      // Also delete associated comments
      setComments((prevComments) =>
        prevComments.filter((comment) => comment.task !== taskId)
      );

      toast({
        title: "Task Deleted",
        description: "The task has been successfully deleted.",
      });

      return true;
    } catch (error) {
      console.error("Error deleting task:", error);
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to delete task. Please try again.",
      });
      return false;
    } finally {
      setLoading(false);
    }
  };

  // Add a comment to a task
  const addComment = async (taskId, text) => {
    try {
      setLoading(true);

      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 300));

      const task = tasks.find((task) => task.id === taskId);

      if (!task) {
        throw new Error("Task not found");
      }

      const newComment = {
        id: Date.now(),
        taskId,
        user,
        text,
        created_at: new Date().toISOString(),
      };

      setComments((prevComments) => [newComment, ...prevComments]);

      // Check for mentions in the comment
      const mentionRegex = /@(\w+\s\w+)/g;
      const mentions = text.match(mentionRegex);

      if (mentions) {
        mentions.forEach((mention) => {
          const username = mention.substring(1); // Remove @ symbol
          // In a real app, you would look up the user ID based on the name
          // For now, we'll use simplified logic
          if (username === task.assigned_to.name) {
            addNotification({
              userId: task.assigned_to.id,
              text: `${user.username} mentioned you in a comment on task "${task.title}"`,
            });
          } else if (username === task.created_by.name) {
            addNotification({
              userId: task.created_by.id,
              text: `${user.username} mentioned you in a comment on task "${task.title}"`,
            });
          }
        });
      }

      toast({
        title: "Comment Added",
        description: "Your comment has been added to the task.",
      });

      return newComment;
    } catch (error) {
      console.error("Error adding comment:", error);
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to add comment. Please try again.",
      });
      return null;
    } finally {
      setLoading(false);
    }
  };

  // Add a notification
  const addNotification = (notificationData) => {
    const newNotification = {
      id: Date.now(),
      ...notificationData,
      isRead: false,
      created_at: new Date().toISOString(),
    };

    setNotifications((prevNotifications) => [
      newNotification,
      ...prevNotifications,
    ]);
    return newNotification;
  };

  // Mark notification as read
  const markNotificationAsRead = (notificationId) => {
    setNotifications((prevNotifications) =>
      prevNotifications.map((notification) =>
        notification.id === notificationId
          ? { ...notification, isRead: true }
          : notification
      )
    );
  };

  // Get user's tasks
  const getUserTasks = (userId) => {
    return tasks.filter((task) => task.assigned_to.id === userId);
  };

  // Get task by id
  const getTaskById = (taskId) => {
    return tasks.find((task) => task.id === taskId) || null;
  };

  // Get comments for a task
  const getTaskComments = (taskId) => {
    return comments.filter((comment) => comment.task === taskId);
  };

  // Get user's notifications
  const getUserNotifications = (userId) => {
    return notifications.filter(
      (notification) => notification.userId === userId
    );
  };

  // Calculate task statistics
  const getTaskStats = () => {
    const totalTasks = tasks.length;
    const completedTasks = tasks.filter(
      (task) => task.status === "completed"
    ).length;
    const inProgressTasks = tasks.filter(
      (task) => task.status === "in_progress"
    ).length;
    const pendingTasks = tasks.filter(
      (task) => task.status === "pending"
    ).length;

    const highPriority = tasks.filter(
      (task) => task.priority === "high"
    ).length;
    const mediumPriority = tasks.filter(
      (task) => task.priority === "medium"
    ).length;
    const lowPriority = tasks.filter((task) => task.priority === "low").length;

    return {
      totalTasks,
      completedTasks,
      inProgressTasks,
      pendingTasks,
      highPriority,
      mediumPriority,
      lowPriority,
      completionRate: totalTasks > 0 ? (completedTasks / totalTasks) * 100 : 0,
    };
  };

  return (
    <TaskContext.Provider
      value={{
        tasks,
        comments,
        notifications,
        loading,
        addTask,
        updateTask,
        deleteTask,
        addComment,
        markNotificationAsRead,
        getUserTasks,
        getTaskById,
        getTaskComments,
        getUserNotifications,
        getTaskStats,
      }}
    >
      {children}
    </TaskContext.Provider>
  );
};

export default TaskContext;
