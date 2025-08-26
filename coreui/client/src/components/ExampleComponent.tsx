type ExampleComponentProps = {
  title: string;
};

export function ExampleComponent({ title }: ExampleComponentProps) {
  return <div className="text-2xl">{title}</div>;
}
