import { ShellLayout } from "./layouts/ShellLayout";
import { WorkspaceLayout } from "./layouts/WorkspaceLayout";

/* CC-BY-SA 4.0 notice: the desktop proof-surface presentation layer, consumer
   copy, and immersive visualization layout in this file are licensed for
   share-alike reuse where the repository documents that UI-specific split. */

export default function App() {
  return (
    <ShellLayout>
      <WorkspaceLayout />
    </ShellLayout>
  );
}
