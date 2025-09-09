type Tag = {
  name: string;
};

export function TagCard({ key, tag }: { key: number; tag: Tag }) {
  return (
    <div
      key={key}
      className="border-2 border-black border-solid rounded-sm shadow-md m-2 p-2"
    >
      <label>
        <span className="block text-xs font-bold">Tag Name</span>
        <input
          type="text"
          value={tag.name}
          className="border-2 border-black border-solid rounded-sm text-base pl-1"
        />
      </label>
    </div>
  );
}
