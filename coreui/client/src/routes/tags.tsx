import { createFileRoute } from '@tanstack/react-router';
import { useQuery } from '@tanstack/react-query';
import { ExampleComponent } from '../components/ExampleComponent';
import { TagCard } from '../components/navigation/Tags'

export const Route = createFileRoute('/tags')({
  component: Tags,
});

type Tag = {
    name: string;
}

function Tags() {
  const { isPending, error, data } = useQuery({
    queryKey: ['tags'],
    queryFn: () =>
        fetch('http://localhost:8000/api/configs/main_backwash/tags'
    ).then((res) => 
        res.json(),
    ),
  });

  if (isPending) return 'Loading tags..'

  if (error) return ' :-( ' + error.message

  return (
    <div className="p-2">
      <ExampleComponent title="Tags" />
      { data.tags.map((tag: Tag, index: number) => {
        return <TagCard key={index} tag={tag} />
      }) }
    </div>
  );
}
