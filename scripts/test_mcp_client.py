import asyncio
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

async def main():
    server_params = StdioServerParameters(
        command=r'C:\Users\Jy-Mentor-7\anaconda3\python.exe',
        args=['-m', 'paper_search_mcp.server'],
        env=None,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            result = await session.initialize()
            print('initialize result:', result)

            tools = await session.list_tools()
            print('tools count:', len(tools.tools))
            print('first 5 tools:', [t.name for t in tools.tools[:5]])

if __name__ == '__main__':
    asyncio.run(main())
