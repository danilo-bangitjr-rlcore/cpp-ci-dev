import { createFileRoute } from '@tanstack/react-router';
import { useQuery } from '@tanstack/react-query';
import { ExampleComponent } from '../components/ExampleComponent';
import { TagCard } from '../components/tags-components/Tags'
import { API_ENDPOINTS, get } from '../utils/api';

export const Route = createFileRoute('/tags')({
  component: Tags,
});

type Tag = {
    name: string;
}

function Tags() {
  const { isPending, error, data } = useQuery({
    queryKey: ['tags'],
    queryFn: async () => {
      const response = await get(API_ENDPOINTS.configs.tags);
      if (!response.ok) throw new Error('Failed to fetch tags');
      return response.json();
    },
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
