import argparse

# ArgumentParser 객체 생성
parser = argparse.ArgumentParser(description='간단한 계산기 프로그램')

# 인자 추가
parser.add_argument('operation', choices=['add', 'subtract', 'multiply', 'divide'], help='수행할 연산')
parser.add_argument('x', type=float, help='첫 번째 숫자')
parser.add_argument('y', type=float, help='두 번째 숫자')

# 인자 파싱
args = parser.parse_args()

# 연산 수행
if args.operation == 'add':
    result = args.x + args.y
elif args.operation == 'subtract':
    result = args.x - args.y
elif args.operation == 'multiply':
    result = args.x * args.y
elif args.operation == 'divide':
    result = args.x / args.y

# 결과 출력
print(f"결과: {result}")
