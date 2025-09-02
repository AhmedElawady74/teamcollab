import { useContext, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import { useTask } from "@/context/TaskContext";
import { Bell, Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import AuthContext from "../context/AuthContext";

const Navbar = () => {
  const { user, logoutUser } = useContext(AuthContext);
  const { notifications, markNotificationAsRead } = useTask();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const navigate = useNavigate();

  // Filter notifications for the current user
  const userNotifications = user
    ? notifications.filter((notif) => notif.userId === user.id)
    : [];

  const unreadCount = userNotifications.filter((notif) => !notif.isRead).length;

  const handleNotificationClick = (notificationId) => {
    markNotificationAsRead(notificationId);
  };

  const formatNotificationTime = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 60) {
      return `${diffMins}m ago`;
    } else if (diffHours < 24) {
      return `${diffHours}h ago`;
    } else {
      return `${diffDays}d ago`;
    }
  };

  return (
    <nav className="bg-white shadow-sm border-b">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center">
              <span
                onClick={() => navigate("/")}
                className="cursor-pointer text-2xl font-bold bg-clip-text text-transparent hive-gradient"
              >
                TaskHive
              </span>
            </div>

            {/* Desktop Navigation */}
            <div className="hidden md:ml-6 md:flex md:space-x-8">
              {user && (
                <>
                  <button
                    onClick={() => navigate("/dashboard")}
                    className="border-transparent text-gray-600 hover:border-primary hover:text-primary inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium"
                  >
                    Dashboard
                  </button>
                  <button
                    onClick={() => navigate("/tasks")}
                    className="border-transparent text-gray-600 hover:border-primary hover:text-primary inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium"
                  >
                    Tasks
                  </button>
                  {user.role === "manager" && (
                    <button
                      onClick={() => navigate("/task-management")}
                      className="border-transparent text-gray-600 hover:border-primary hover:text-primary inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium"
                    >
                      Manage Tasks
                    </button>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Right side navigation items */}
          <div className="flex items-center">
            {user ? (
              <>
                {/* Notification Bell */}
                <div className="ml-3 relative">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon" className="relative">
                        <Bell className="h-5 w-5" />
                        {unreadCount > 0 && (
                          <span className="absolute top-0 right-0 h-4 w-4 rounded-full bg-red-500 text-xs flex items-center justify-center text-white">
                            {unreadCount}
                          </span>
                        )}
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-80">
                      <DropdownMenuLabel>Notifications</DropdownMenuLabel>
                      <DropdownMenuSeparator />
                      {userNotifications.length > 0 ? (
                        userNotifications.slice(0, 5).map((notification) => (
                          <DropdownMenuItem
                            key={notification.id}
                            onClick={() =>
                              handleNotificationClick(notification.id)
                            }
                            className={`flex flex-col items-start cursor-pointer ${
                              !notification.isRead ? "bg-blue-50" : ""
                            }`}
                          >
                            <div className="flex justify-between w-full">
                              <p className="text-sm">{notification.text}</p>
                              {!notification.isRead && (
                                <Badge
                                  variant="outline"
                                  className="ml-2 h-2 w-2 bg-blue-500 rounded-full"
                                />
                              )}
                            </div>
                            <span className="text-xs text-gray-500 mt-1">
                              {formatNotificationTime(notification.createdAt)}
                            </span>
                          </DropdownMenuItem>
                        ))
                      ) : (
                        <div className="text-center py-4 text-sm text-gray-500">
                          No notifications
                        </div>
                      )}
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>

                {/* User Menu */}
                <div className="ml-3 relative">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="ghost"
                        className="flex items-center space-x-2"
                      >
                        <Avatar className="h-8 w-8">
                          <AvatarFallback>
                            {user.username.charAt(0)}
                          </AvatarFallback>
                        </Avatar>
                        <span className="hidden md:block text-sm font-medium">
                          {user.username}
                        </span>
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuLabel>
                        <div className="flex flex-col">
                          <span>{user.username}</span>
                          <span className="text-xs text-gray-500">
                            {user.role}
                          </span>
                        </div>
                      </DropdownMenuLabel>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        className="cursor-pointer"
                        onClick={() => navigate("/dashboard")}
                      >
                        Dashboard
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        className="cursor-pointer"
                        onClick={() => navigate("/tasks")}
                      >
                        My Tasks
                      </DropdownMenuItem>
                      <DropdownMenuSeparator />
                      <DropdownMenuItem
                        className="cursor-pointer text-red-500 focus:text-red-500"
                        onClick={logoutUser}
                      >
                        Logout
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </>
            ) : (
              <div className="flex space-x-4">
                <Button onClick={() => navigate("/login")}>
                  Log in
                </Button>
                {/* <Button onClick={() => navigate("/signup")}>Register</Button> */}
              </div>
            )}

            {/* Mobile menu button */}
            <div className="md:hidden flex items-center ml-4">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              >
                {mobileMenuOpen ? (
                  <X className="h-6 w-6" />
                ) : (
                  <Menu className="h-6 w-6" />
                )}
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileMenuOpen && user && (
        <div className="md:hidden">
          <div className="pt-2 pb-3 space-y-1">
            <button
              onClick={() => {
                navigate("/dashboard");
                setMobileMenuOpen(false);
              }}
              className="block pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-gray-600 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-800"
            >
              Dashboard
            </button>
            <button
              onClick={() => {
                navigate("/tasks");
                setMobileMenuOpen(false);
              }}
              className="block pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-gray-600 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-800"
            >
              Tasks
            </button>
            {user.role === "manager" && (
              <button
                onClick={() => {
                  navigate("/task-management");
                  setMobileMenuOpen(false);
                }}
                className="block pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-gray-600 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-800"
              >
                Manage Tasks
              </button>
            )}
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
