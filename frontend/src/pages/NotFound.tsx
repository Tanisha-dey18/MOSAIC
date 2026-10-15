import { Link } from "react-router-dom";
import { EmptyState } from "../components/ui";

export default function NotFound() {
  return (
    <EmptyState title="Page not found" action={<Link className="btn-primary" to="/">Back to dashboard</Link>}>
      The page you requested does not exist.
    </EmptyState>
  );
}
