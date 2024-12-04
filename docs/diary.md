The first point is to implement a parser for the input task and its context.

The parse is done.

The another part is to create a storage for task solutions. I will not create any methods for storing, so the storage will be always empty for now.

4.12.2024
I have set up the solution proposal. Now I need to implement the solution storage. When retrieving a solution, I will use a text query containing approximate description of the task, the solution should be used for.
That's why I will use a vector database.

There will actually be two databases:
- Solution database. This contains solutions related to specific tasks in given context.
- Source database. This contains information retrieved during solving various tasks. Each source is related to a context and question/query.