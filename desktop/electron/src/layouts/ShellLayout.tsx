import type { ReactNode } from "react";
import { DesktopAppProvider } from "../state/DesktopAppContext";
import NavSidebar from "./NavSidebar";

type ShellLayoutProps = {
  children: ReactNode;
};

export function ShellLayout({ children }: ShellLayoutProps) {
  return (
    <DesktopAppProvider>
      <div className="shell">
        <NavSidebar />
        <main className="workspace">{children}</main>
      </div>
    </DesktopAppProvider>
  );
}
