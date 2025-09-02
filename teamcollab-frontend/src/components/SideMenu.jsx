
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { LayoutDashboard, ListChecks, Settings, Users } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { USER_ROLES } from '@/lib/constants';

const SideMenu = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();
  
  if (!user) return null;
  
  const menuItems = [
    {
      name: 'Dashboard',
      icon: <LayoutDashboard className="h-5 w-5" />,
      path: '/dashboard',
      roles: [USER_ROLES.MANAGER, USER_ROLES.DEVELOPER, USER_ROLES.TESTER],
    },
    {
      name: 'Tasks',
      icon: <ListChecks className="h-5 w-5" />,
      path: '/tasks',
      roles: [USER_ROLES.MANAGER, USER_ROLES.DEVELOPER, USER_ROLES.TESTER],
    },
    {
      name: 'Task Management',
      icon: <Settings className="h-5 w-5" />,
      path: '/task-management',
      roles: [USER_ROLES.MANAGER],
    },
  ];
  
  const filteredItems = menuItems.filter(
    item => item.roles.includes(user.role)
  );
  
  return (
    <div className="w-64 bg-white border-r min-h-screen p-4">
      <div className="mb-8">
        <h2 className="text-xl font-bold mb-1 bg-clip-text text-transparent hive-gradient">TaskHive</h2>
        <p className="text-sm text-gray-500">{user.name}</p>
        <p className="text-xs text-gray-500">{user.role}</p>
      </div>
      
      <div className="space-y-1">
        {filteredItems.map((item) => (
          <Button
            key={item.path}
            variant={location.pathname === item.path ? "default" : "ghost"}
            className="w-full justify-start"
            onClick={() => navigate(item.path)}
          >
            {item.icon}
            <span className="ml-2">{item.name}</span>
          </Button>
        ))}
      </div>
    </div>
  );
};

export default SideMenu;
