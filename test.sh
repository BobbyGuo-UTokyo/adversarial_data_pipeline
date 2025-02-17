curl -v https://ark.cn-beijing.volces.com/api/v3/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 0a46cc4e-7bcd-4a50-a32f-858048a9a976" \
  -d '{
    "model": "ep-20250216235228-69vhs",
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
