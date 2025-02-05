import { CheckIcon } from "@heroicons/react/24/solid";
import { Link } from "./link";
import { LinkComponentProps } from "@tanstack/react-router";
import { Text } from "./text";

export interface Step {
    name: string
    to: LinkComponentProps["to"]
    status: "complete" | "current" | "upcoming"
}

export function ProgressBar({
  className,
  steps,
  ...props
}: React.ComponentPropsWithoutRef<"nav"> & { steps: Step[] }) {
  return (
    <nav aria-label="Progress" {...props} className={className}>
      <ol
        role="list"
        className="divide-y divide-gray-300 rounded-md border border-gray-300 md:flex md:divide-y-0"
      >
        {steps.map((step, stepIdx) => (
          <li key={step.name} className="relative md:flex md:flex-1">
            {step.status === "complete" ? (
              <Link to={step.to} className="group flex w-full items-center">
                <span className="flex items-center px-6 py-4 text-sm font-medium">
                  <span className="flex size-10 shrink-0 items-center justify-center rounded-full bg-indigo-600 group-hover:bg-indigo-800">
                    <CheckIcon
                      aria-hidden="true"
                      className="size-6 text-white"
                    />
                  </span>
                  <Text className="ml-4">
                    {step.name}
                  </Text>
                </span>
              </Link>
            ) : step.status === "current" ? (
              <Link
                to={step.to}
                aria-current="step"
                className="flex items-center px-6 py-4 text-sm font-medium"
              >
                <span className="flex size-10 shrink-0 items-center justify-center rounded-full border-2 border-indigo-600">
                  <Text className="text-indigo-600">{stepIdx}</Text>
                </span>
                <Text className="ml-4">
                  {step.name}
                </Text>
              </Link>
            ) : (
              <Link to={step.to} className="group flex items-center">
                <span className="flex items-center px-6 py-4 text-sm font-medium">
                  <span className="flex size-10 shrink-0 items-center justify-center rounded-full border-2 border-gray-300 group-hover:border-gray-400">
                    <Text className="text-gray-500 group-hover:text-gray-900">
                      {stepIdx}
                    </Text>
                  </span>
                  <Text className="ml-4">
                    {step.name}
                  </Text>
                </span>
              </Link>
            )}

            {stepIdx !== steps.length - 1 ? (
              <>
                {/* Arrow separator for lg screens and up */}
                <div
                  aria-hidden="true"
                  className="absolute top-0 right-0 hidden h-full w-5 md:block"
                >
                  <svg
                    fill="none"
                    viewBox="0 0 22 80"
                    preserveAspectRatio="none"
                    className="size-full text-gray-300"
                  >
                    <path
                      d="M0 -2L20 40L0 82"
                      stroke="currentcolor"
                      vectorEffect="non-scaling-stroke"
                      strokeLinejoin="round"
                    />
                  </svg>
                </div>
              </>
            ) : null}
          </li>
        ))}
      </ol>
    </nav>
  );
}
