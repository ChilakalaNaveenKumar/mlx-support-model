# users = [{
#     "name": "Naveen",
#     "skills": ["Python", "Django", "Vue.js"],
#     "years_experience": 6
# }, {
#     "name": "John",
#     "skills": ["Python", "Vue.js"],
#     "years_experience": 6
# }, {
#     "name": "Jane",
#     "skills": ["Vue.js"],
#     "years_experience": 6
# }]

class User:
    def __init__(self, name, skills=None, years_experience=0):
        self.name = name
        self.skills = skills or []
        self.years_experience = years_experience
    
    def add_skill(self, skill):
        if skill not in self.skills:
            self.skills.append(skill)
    
    def has_skill(self, skill):
        return skill in self.skills
    def get_experience(self):
        return self.years_experience
    
    def __str__(self):
        return f"{self.name} ({len(self.skills)} skills, {self.years_experience} years)"
    
users = []

users.append(User(name = "Naveen", skills = ["Python", "Django", "Vue.js"], years_experience = 6))
users.append(User(name = "John", skills = ["Python", "Django", "Vue.js"], years_experience = 7))
users.append(User(name = "Jane", skills = ["Vue.js"], years_experience = 8))


def get_most_experinced_python(all_users):
    experience = 0
    skill = "Python"
    experineced_user = ''
    for user in all_users:
        print(user.has_skill(skill) , user.get_experience(), user.has_skill(skill) & experience < user.get_experience())
        if user.has_skill(skill) & experience < user.get_experience():
            experineced_user = user.__str__()
    print(experineced_user)

get_most_experinced_python(users)