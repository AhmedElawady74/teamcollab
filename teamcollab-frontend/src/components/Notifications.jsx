
import { useEffect } from 'react';
import { useTask } from '@/context/TaskContext';
import { useAuth } from '@/context/AuthContext';
import { Bell } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

const Notifications = () => {
  const { user } = useAuth();
  const { getUserNotifications, markNotificationAsRead } = useTask();
  
  const userNotifications = user ? getUserNotifications(user.id) : [];
  const unreadCount = userNotifications.filter(notif => !notif.isRead).length;
  
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
  
  const handleNotificationClick = (notificationId) => {
    markNotificationAsRead(notificationId);
  };
  
  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg font-semibold flex items-center">
            <Bell className="h-5 w-5 mr-2" />
            Notifications
            {unreadCount > 0 && (
              <Badge className="ml-2 bg-primary">{unreadCount}</Badge>
            )}
          </CardTitle>
          {unreadCount > 0 && (
            <Button 
              variant="ghost" 
              size="sm"
              onClick={() => userNotifications.forEach(n => !n.isRead && markNotificationAsRead(n.id))}
            >
              Mark all as read
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[300px]">
          {userNotifications.length > 0 ? (
            <div className="space-y-3">
              {userNotifications.map((notification) => (
                <div 
                  key={notification.id} 
                  className={`p-3 rounded-md cursor-pointer ${!notification.isRead ? 'bg-blue-50' : 'bg-gray-50'}`}
                  onClick={() => handleNotificationClick(notification.id)}
                >
                  <div className="flex justify-between">
                    <p className="text-sm">{notification.text}</p>
                    {!notification.isRead && (
                      <Badge variant="outline" className="h-2 w-2 bg-blue-500 rounded-full" />
                    )}
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    {formatNotificationTime(notification.createdAt)}
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <p>No notifications</p>
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
};

export default Notifications;
