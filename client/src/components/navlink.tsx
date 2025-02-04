import { Link, LinkComponentProps } from "@tanstack/react-router";

const NavLink = ({ to, children, ...props }: LinkComponentProps) => (
  <Link
    to={to}
    className="block px-4 py-2 hover:bg-gray-200"
    activeOptions={{ exact: true }}
    activeProps={{ className: "bg-gray-200" }}
    {...props}
  >
    {children}
  </Link>
);

export default NavLink;
