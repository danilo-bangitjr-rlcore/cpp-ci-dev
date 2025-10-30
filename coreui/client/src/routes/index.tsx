import { createFileRoute } from '@tanstack/react-router';

export const Route = createFileRoute('/')({
  component: Index,
});

function Index() {
  return (
    <div className="h-full w-full">
      <iframe
        src="https://www.rlcore.ai"
        className="w-full h-full border-0"
        title="RLCore Website"
      />
    </div>
  );
}
