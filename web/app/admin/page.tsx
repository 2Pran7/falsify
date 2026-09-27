import { AdminClient } from "./AdminClient";

export const metadata = { title: "Admin · falsify", robots: { index: false, follow: false } };

export default function AdminPage() {
  return <AdminClient />;
}
