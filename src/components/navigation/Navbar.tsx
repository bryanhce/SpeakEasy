"use client";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Home, LogOut, User } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useUserAuth } from "../auth/UserAuthContext";
import SnapPhoto from "./SnapPhoto";

const menuItems = [{ name: "Dashboard", icon: <Home size={14} /> }];

export default function Navbar() {
  const { logOut } = useUserAuth();
  const navigate = useNavigate();

  const logOutUser = () => {
    logOut();
    navigate("/login");
  };

  const DesktopNav = () => (
    <nav className="hidden md:flex items-center justify-between bg-background p-4 border-b">
      <div className="flex items-center space-x-4">
        <a href="/" className="text-2xl font-bold">
          <img src="/logo.svg" alt="logo" className="h-10" />
        </a>
        {menuItems.map((item) => (
          <a
            key={item.name}
            href={`/${item.name.toLowerCase()}`}
            className="text-foreground hover:text-primary"
          >
            {item.name}
          </a>
        ))}
      </div>
      <div className="flex items-center space-x-4">
        <SnapPhoto isMobile={false} />
        <ProfileDropdown />
      </div>
    </nav>
  );

  const MobileNav = () => (
    <nav className="fixed bottom-0 left-0 right-0 bg-background p-2 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.1)] md:hidden">
      <div className="flex items-center justify-between">
        <div className="flex justify-around flex-1">
          {menuItems.slice(0, 2).map((item) => (
            <a
              key={item.name}
              href={`/${item.name.toLowerCase()}`}
              className="flex flex-col items-center space-y-1 text-foreground hover:text-primary"
            >
              {item.icon}
              <span className="text-xs">{item.name}</span>
            </a>
          ))}
          <SnapPhoto isMobile={true} />
          <ProfileDropdown />
        </div>
      </div>
    </nav>
  );

  const ProfileDropdown = () => (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <div>
          <div className="md:hidden flex flex-col items-center space-y-1 text-foreground hover:text-primary">
            <User size={14} />
            <span className="text-xs">Profile</span>
          </div>
          <button className="hidden md:flex">
            <User size={16} />
          </button>
        </div>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuLabel>My Account</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          onClick={logOutUser}
          className="flex space-x-2 items-center"
        >
          <LogOut size={12} />
          <span>Logout</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );

  return (
    <>
      <DesktopNav />
      <MobileNav />
    </>
  );
}
