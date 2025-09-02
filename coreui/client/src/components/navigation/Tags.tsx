type Tag = {
    name: string;
}

export function TagCard({key, tag}: { key: number, tag: Tag}) {
    return (
        <div key={key} className=''>
            <label>
                <span className='block text-sm font-bold'>Tag Name</span>
                <input type='text' value={tag.name} className='border border-black border-solid rounded-sm text-base'/>
            </label>
        </div>
    );
};